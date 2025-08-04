import copy
import logging
import torch
from torch import nn
from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear
import timm
import torch.nn.functional as F
from convs.projections import Proj_Pure_MLP, MultiHeadAttention
from utils.toolkit import get_attribute

def get_convnet(args, pretrained=False):
    """Get convolutional network backbone"""
    backbone_name = args["convnet_type"].lower()
    algorithm_name = args["model_name"].lower()
    if 'clip' in backbone_name:
        print('Using CLIP model as the backbone')
        import open_clip
        if backbone_name == 'clip':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim = 512
            return model, preprocess, tokenizer
        elif backbone_name=='clip_laion2b':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim = 512
            return model, preprocess, tokenizer
        elif backbone_name=='openai_clip':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim = 512
            return model, preprocess, tokenizer
        else:
            raise NotImplementedError("Unknown type {}".format(backbone_name))
    else:
        raise NotImplementedError("Unknown type {}".format(backbone_name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()
        self.convnet = get_convnet(args, pretrained)
        self.fc = None
        self.device = args["device"][0]
        self.to(self.device)

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features,
            'logits': logits
        }
        """
        out.update(x)
        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self


class IncrementalNet(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = (torch.norm(weights[-increment:, :], p=2, dim=1))
        oldnorm = (torch.norm(weights[:-increment, :], p=2, dim=1))
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        out.update(x)
        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None

    def set_gradcam_hook(self):
        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            self._gradcam_backward_hook
        )
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            self._gradcam_forward_hook
        )

    def _gradcam_backward_hook(self, module, grad_input, grad_output):
        self._gradcam_grad = grad_output[0]

    def _gradcam_forward_hook(self, module, input, output):
        self._gradcam_fmap = output


class CosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained, nb_proxy=1):
        super().__init__(args, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.nb_proxy == 1:
            fc = CosineLinear(in_dim, out_dim)
        else:
            fc = SplitCosineLinear(in_dim, out_dim, self.nb_proxy)
        return fc


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_alpha = self.alpha.clamp(low_range, high_range)
        ret_beta = self.beta.clamp(low_range, high_range)
        return x * ret_alpha + ret_beta

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class IncrementalNetWithBias(BaseNet):
    def __init__(self, args, pretrained, bias_correction=False):
        super().__init__(args, pretrained)

        # Increment
        self.increment = 0
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList()

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        if self.bias_correction:
            logits = out["logits"]
            for i, layer in enumerate(self.bias_layers):
                logits = layer(logits, 0, 10)
            out["logits"] = logits
        out.update(x)
        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.increment)
        self.increment.append(new_task_size)
        if self.bias_correction:
            self.bias_layers.append(BiasLayer())

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())
        return params

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc


class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def encode_image(self, x):
        return self.convnet.encode_image(x)

    def encode_text(self, x):
        return self.convnet.encode_text(x)

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        out.update(x)
        return out


class SimpleClipNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.tokenizer = self.convnet[1]

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.convnet[0](x)["features"]

    def encode_image(self, x):
        return self.convnet[0].encode_image(x)

    def encode_text(self, x):
        return self.convnet[0].encode_text(x)

    def forward(self, img, text):
        image_features = self.encode_image(img)
        text_features = self.encode_text(text)
        logits = image_features @ text_features.t()
        return {"logits": logits}

    def re_initiate(self):
        pass


class STAGE_Net(SimpleClipNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.projs_img = nn.ModuleList()
        self.projs_text = nn.ModuleList()
        self.projs_state = nn.ModuleList()
        self.args = args
        self._device = args["device"][0]
        self.projtype = get_attribute(self.args, 'projection_type', 'mlp')
        self.context_prompt_length_per_task = get_attribute(self.args, 'context_prompt_length_per_task', 3)
        
        self.sel_attn = MultiHeadAttention(1, self.feature_dim, self.feature_dim, self.feature_dim, dropout=0.1)
        self.img_prototypes = None
        self.context_prompts = nn.ParameterList()
        self.num_states = 2
        self.state_embedding = nn.Embedding(self.num_states, self.feature_dim)
        
        # Initialize prototype storage structure
        self.img_prototypes_by_state = {}

    def update_prototype(self, nb_classes):
        """Update prototype structure for Stage-CIL"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        if not hasattr(self, "img_prototypes_by_state"):
            self.img_prototypes_by_state = {}
        
        for class_id in range(nb_classes):
            if class_id not in self.img_prototypes_by_state:
                self.img_prototypes_by_state[class_id] = {}
        
        if self.img_prototypes is not None:
            nb_output = len(self.img_prototypes)
            self.img_prototypes = torch.cat([
                copy.deepcopy(self.img_prototypes).to(self._device),
                torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)
            ]).to(self._device)
        else:
            self.img_prototypes = torch.zeros(nb_classes, self.feature_dim).to(self._device)
        
        print(f'Updated prototypes, now have {nb_classes} class prototypes and stage prototype dictionary')
    
    def update_context_prompt(self):
        """Update context prompts for new tasks"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        for i in range(len(self.context_prompts)):
            self.context_prompts[i].requires_grad = False
        self.context_prompts.append(nn.Parameter(torch.randn(self.context_prompt_length_per_task, self.feature_dim).to(self._device)))
        print('Updated context prompt, now we have {} context prompts'.format(len(self.context_prompts) * self.context_prompt_length_per_task))
        self.context_prompts.to(self._device)
    
    def get_context_prompts(self):
        """Get concatenated context prompts"""
        return torch.cat([item for item in self.context_prompts], dim=0)

    def encode_image(self, x, normalize: bool = False):
        """Encode image features"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        x = x.to(self._device)
        basic_img_features = self.convnet.encode_image(x)
        if not self.projs_img:
            logging.warning("encode_image called but no image projections (projs_img) are defined.")
            return F.normalize(basic_img_features, dim=-1) if normalize else basic_img_features

        img_features = [proj(basic_img_features) for proj in self.projs_img]
        img_features = torch.stack(img_features, dim=1)
        image_feas = torch.sum(img_features, dim=1)
        return F.normalize(image_feas, dim=-1) if normalize else image_feas
        
    def encode_text(self, x, normalize: bool = False):
        """Encode text features"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        x = x.to(self._device)
        basic_text_features = self.convnet.encode_text(x)
        if not self.projs_text:
            logging.warning("encode_text called but no text projections (projs_text) are defined.")
            return F.normalize(basic_text_features, dim=-1) if normalize else basic_text_features

        text_features = [proj(basic_text_features) for proj in self.projs_text]
        text_features = torch.stack(text_features, dim=1)
        text_feas = torch.sum(text_features, dim=1)
        return F.normalize(text_feas, dim=-1) if normalize else text_feas
        
    def encode_prototypes(self, normalize: bool = False):
        """Encode prototype features"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        self.img_prototypes = self.img_prototypes.to(self._device)
        if not self.projs_img:
            logging.warning("encode_prototypes called but no image projections (projs_img) are defined.")
            return F.normalize(self.img_prototypes, dim=-1) if normalize else self.img_prototypes

        img_features = [proj(self.img_prototypes) for proj in self.projs_img]
        img_features = torch.stack(img_features, dim=1)
        image_feas = torch.sum(img_features, dim=1)
        return F.normalize(image_feas, dim=-1) if normalize else image_feas
    
    def encode_stage_prototypes(self, class_id, stage_id, normalize: bool = False):
        """Encode stage-specific prototype features - Core Stage-CIL functionality"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        if (class_id not in self.img_prototypes_by_state or 
            stage_id not in self.img_prototypes_by_state[class_id]):
            logging.warning(f"No prototype found for class {class_id}, stage {stage_id}")
            return None
            
        stage_proto = self.img_prototypes_by_state[class_id][stage_id].to(self._device)
        
        if not self.projs_img:
            logging.warning("encode_stage_prototypes called but no image projections defined.")
            return F.normalize(stage_proto, dim=-1) if normalize else stage_proto
            
        stage_features = [proj(stage_proto.unsqueeze(0)) for proj in self.projs_img]
        stage_features = torch.stack(stage_features, dim=1)
        stage_feas = torch.sum(stage_features, dim=1).squeeze(0)
        return F.normalize(stage_feas, dim=-1) if normalize else stage_feas

    def extend_task(self):
        """Extend task-specific projections"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        self.projs_img.append(self.extend_item())
        self.projs_text.append(self.extend_item())
        self.projs_state.append(self.extend_item())
        print(f"Task extension: Added new projections, now have {len(self.projs_img)} sets of three-way projections")

    def extend_item(self):
        """Create new projection item"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        if self.projtype == 'pure_mlp':
            return Proj_Pure_MLP(self.feature_dim, self.feature_dim, self.feature_dim).to(self._device)
        else:
            raise NotImplementedError
    
    def forward(self, image, text):
        """Forward pass with multi-modal attention"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        prototype_features = self.encode_prototypes(normalize=True)
        context_prompts = self.get_context_prompts()

        len_texts = text_features.shape[0]
        len_protos = prototype_features.shape[0]
        len_context_prompts = context_prompts.shape[0]
        image_features = image_features.view(image_features.shape[0], -1, self.feature_dim)
        text_features = text_features.view(text_features.shape[0], self.feature_dim)
        prototype_features = prototype_features.view(prototype_features.shape[0], self.feature_dim)
        context_prompts = context_prompts.view(context_prompts.shape[0], self.feature_dim)
        text_features = text_features.expand(image_features.shape[0], text_features.shape[0], self.feature_dim)
        prototype_features = prototype_features.expand(image_features.shape[0], prototype_features.shape[0], self.feature_dim)
        context_prompts = context_prompts.expand(image_features.shape[0], context_prompts.shape[0], self.feature_dim)
        features = torch.cat([image_features, text_features, prototype_features, context_prompts], dim=1)
        features = self.sel_attn(features, features, features)
        image_features = features[:, 0, :]
        text_features = features[:, 1:len_texts+1, :]
        prototype_features = features[:, len_texts+1:len_texts+1+len_protos, :]
        context_prompts = features[:, len_texts+1+len_protos:len_texts+1+len_protos+len_context_prompts, :]
        text_features = torch.mean(text_features, dim=0)
        prototype_features = torch.mean(prototype_features, dim=0)
        image_features = image_features.view(image_features.shape[0], -1)
        text_features = text_features.view(text_features.shape[0], -1)
        prototype_features = prototype_features.view(prototype_features.shape[0], -1)
        return image_features, text_features, self.convnet.logit_scale.exp(), prototype_features
    
    def forward_transformer(self, image_features, text_features, transformer=False):
        """Forward pass with transformer attention"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        prototype_features = self.encode_prototypes(normalize=True)
        if transformer:
            context_prompts = self.get_context_prompts()
            len_texts = text_features.shape[0]
            len_protos = prototype_features.shape[0]
            len_context_prompts = context_prompts.shape[0]
            image_features = image_features.view(image_features.shape[0], -1, self.feature_dim)
            text_features = text_features.view(text_features.shape[0], self.feature_dim)
            prototype_features = prototype_features.view(prototype_features.shape[0], self.feature_dim)
            context_prompts = context_prompts.view(context_prompts.shape[0], self.feature_dim)
            text_features = text_features.expand(image_features.shape[0], text_features.shape[0], self.feature_dim)
            prototype_features = prototype_features.expand(image_features.shape[0], prototype_features.shape[0], self.feature_dim)
            context_prompts = context_prompts.expand(image_features.shape[0], context_prompts.shape[0], self.feature_dim)
            features = torch.cat([image_features, text_features, prototype_features, context_prompts], dim=1)
            features = self.sel_attn(features, features, features)
            image_features = features[:, 0, :]
            text_features = features[:, 1:len_texts+1, :]
            prototype_features = features[:, len_texts+1:len_texts+1+len_protos, :]
            context_prompts = features[:, len_texts+1+len_protos:len_texts+1+len_protos+len_context_prompts, :]
            text_features = torch.mean(text_features, dim=0)
            prototype_features = torch.mean(prototype_features, dim=0)
            image_features = image_features.view(image_features.shape[0], -1)
            text_features = text_features.view(text_features.shape[0], -1)
            prototype_features = prototype_features.view(prototype_features.shape[0], -1)
            return image_features, text_features, self.convnet.logit_scale.exp(), prototype_features
        else:
            return image_features, text_features, self.convnet.logit_scale.exp(), prototype_features
    
    def freeze_projection_weight_new(self):
        """Freeze old projection weights, keep new ones trainable"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        if len(self.projs_img) > 1:
            for i in range(len(self.projs_img) - 1):
                for param in self.projs_img[i].parameters():
                    param.requires_grad = False
                for param in self.projs_text[i].parameters():
                    param.requires_grad = False
                for param in self.projs_state[i].parameters():
                    param.requires_grad = False
            
            for param in self.projs_img[-1].parameters():
                param.requires_grad = True
            for param in self.projs_text[-1].parameters():
                param.requires_grad = True
            for param in self.projs_state[-1].parameters():
                param.requires_grad = True
        
        for param in self.sel_attn.parameters():
            param.requires_grad = True
        
        if self.projs_state:
            for param in self.projs_state[-1].parameters():
                param.requires_grad = True

    def encode_state(self, state_ids, normalize: bool = False):
        """Encode state embeddings"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        state_ids = torch.clamp(state_ids, 0, self.num_states - 1)
        state_features = self.state_embedding(state_ids)
        
        if not self.projs_state:
            logging.warning("encode_state called but no state projections (projs_state) are defined.")
            return F.normalize(state_features, dim=-1) if normalize else state_features

        state_features = [proj(state_features) for proj in self.projs_state]
        state_features = torch.stack(state_features, dim=1)
        state_feas = torch.sum(state_features, dim=1)
        return F.normalize(state_feas, dim=-1) if normalize else state_feas

    def forward_tri_modal(self, image, text, state_ids):
        """Forward pass with tri-modal inputs (image, text, state)"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        state_features = self.encode_state(state_ids, normalize=True)
        prototype_features = self.encode_prototypes(normalize=True)
        context_prompts = self.get_context_prompts()

        len_texts = text_features.shape[0]
        len_protos = prototype_features.shape[0]
        len_context_prompts = context_prompts.shape[0]
        len_states = state_features.shape[0]
        
        image_features = image_features.view(image_features.shape[0], -1, self.feature_dim)
        text_features = text_features.view(text_features.shape[0], self.feature_dim)
        state_features = state_features.view(state_features.shape[0], self.feature_dim)
        prototype_features = prototype_features.view(prototype_features.shape[0], self.feature_dim)
        context_prompts = context_prompts.view(context_prompts.shape[0], self.feature_dim)
        
        text_features = text_features.expand(image_features.shape[0], text_features.shape[0], self.feature_dim)
        state_features = state_features.expand(image_features.shape[0], state_features.shape[0], self.feature_dim)
        prototype_features = prototype_features.expand(image_features.shape[0], prototype_features.shape[0], self.feature_dim)
        context_prompts = context_prompts.expand(image_features.shape[0], context_prompts.shape[0], self.feature_dim)
        
        features = torch.cat([image_features, text_features, state_features, prototype_features, context_prompts], dim=1)
        features = self.sel_attn(features, features, features)
        
        image_features = features[:, 0, :]
        text_features = features[:, 1:len_texts+1, :]
        state_features = features[:, len_texts+1:len_texts+1+len_states, :]
        prototype_features = features[:, len_texts+1+len_states:len_texts+1+len_states+len_protos, :]
        context_prompts = features[:, len_texts+1+len_states+len_protos:len_texts+1+len_states+len_protos+len_context_prompts, :]
        
        text_features = torch.mean(text_features, dim=0)
        state_features = torch.mean(state_features, dim=0)
        prototype_features = torch.mean(prototype_features, dim=0)
        
        image_features = image_features.view(image_features.shape[0], -1)
        text_features = text_features.view(text_features.shape[0], -1)
        state_features = state_features.view(state_features.shape[0], -1)
        prototype_features = prototype_features.view(prototype_features.shape[0], -1)
        
        return image_features, text_features, state_features, self.convnet.logit_scale.exp(), prototype_features

    def get_all_stage_prototypes_for_class(self, class_id, normalize: bool = False):
        """Get all stage prototypes for a specific class"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        if class_id not in self.img_prototypes_by_state:
            return None
            
        stage_prototypes = {}
        for stage_id in self.img_prototypes_by_state[class_id]:
            stage_proto = self.encode_stage_prototypes(class_id, stage_id, normalize)
            if stage_proto is not None:
                stage_prototypes[stage_id] = stage_proto
                
        return stage_prototypes

    def update_stage_prototype(self, class_id, stage_id, prototype_tensor):
        """Update stage-specific prototype for a class"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        if class_id not in self.img_prototypes_by_state:
            self.img_prototypes_by_state[class_id] = {}
            
        self.img_prototypes_by_state[class_id][stage_id] = prototype_tensor.cpu()
        logging.info(f"Updated stage prototype for class {class_id}, stage {stage_id}")

    def get_stage_evolution_pairs(self, stage_map={0: 1, 1: None}):
        """Get stage evolution pairs for Stage-CIL"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        evolution_pairs = {}
        for class_id in self.img_prototypes_by_state:
            class_pairs = {}
            for stage_0, stage_1 in stage_map.items():
                if (stage_0 in self.img_prototypes_by_state[class_id] and 
                    stage_1 is not None and stage_1 in self.img_prototypes_by_state[class_id]):
                    class_pairs[stage_0] = stage_1
            if class_pairs:
                evolution_pairs[class_id] = class_pairs
        return evolution_pairs

    def forward_stage_cil(self, image, text, state_ids, class_ids=None):
        """Forward pass specifically for Stage-CIL"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        state_features = self.encode_state(state_ids, normalize=True)
        
        # Stage-specific prototype encoding
        if class_ids is not None:
            stage_prototypes = []
            for class_id in class_ids:
                class_protos = self.get_all_stage_prototypes_for_class(class_id.item(), normalize=True)
                if class_protos:
                    stage_protos = torch.stack(list(class_protos.values()))
                    stage_prototypes.append(stage_protos)
            
            if stage_prototypes:
                prototype_features = torch.cat(stage_prototypes, dim=0)
            else:
                prototype_features = self.encode_prototypes(normalize=True)
        else:
            prototype_features = self.encode_prototypes(normalize=True)
        
        context_prompts = self.get_context_prompts()
        
        # Multi-modal attention processing
        # [ATTENTION PROCESSING DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        
        return image_features, text_features, state_features, prototype_features


