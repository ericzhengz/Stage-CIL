num_workers = 8

class MorphologyDataCollector:
    """Data collector for morphology evolution visualization"""
    
    def __init__(self):
        self.task_results = {}
        self.feature_data = {}
        self.distance_metrics = {}
        self.evolution_analysis = {}
        self.feature_clusters = {}
        
    def collect_task_results(self, task_idx, morphology_stage, pair_idx, 
                           accuracy_metrics, feature_data, distance_metrics):
        """Collect results for a single task"""
        if task_idx not in self.task_results:
            self.task_results[task_idx] = {}
            
        stage_key = f"stage_{morphology_stage}"
        self.task_results[task_idx][stage_key] = {
            'pair_idx': pair_idx,
            'accuracy_metrics': accuracy_metrics,
            'feature_data': feature_data,
            'distance_metrics': distance_metrics
        }
        
    def collect_evolution_analysis(self, task_idx, pair_idx, 
                                 morph0_features, morph1_features,
                                 morph0_prototypes, morph1_prototypes):
        """Collect evolution analysis data"""
        if task_idx not in self.evolution_analysis:
            self.evolution_analysis[task_idx] = {}
            
        self.evolution_analysis[task_idx][pair_idx] = {
            'morph0_features': morph0_features,
            'morph1_features': morph1_features,
            'morph0_prototypes': morph0_prototypes,
            'morph1_prototypes': morph1_prototypes,
            'evolution_metrics': self._compute_evolution_metrics(
                morph0_features, morph1_features, 
                morph0_prototypes, morph1_prototypes
            )
        }
        
    def _compute_evolution_metrics(self, morph0_features, morph1_features,
                                 morph0_prototypes, morph1_prototypes):
        """Compute evolution metrics"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        metrics = {}
        return metrics
        
    def _compute_cluster_stats(self, features):
        """Compute cluster statistics"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        return {}
        
    def save_data(self, save_path):
        """Save all collected data"""
        data_to_save = {
            'task_results': self.task_results,
            'feature_data': self.feature_data,
            'distance_metrics': self.distance_metrics,
            'evolution_analysis': self.evolution_analysis,
            'feature_clusters': self.feature_clusters,
            'metadata': {
                'total_tasks': len(self.task_results),
                'collection_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'data_version': '1.0'
            }
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(data_to_save, f)
            
        logging.info(f"Morphology evolution data saved to: {save_path}")

class LRUCache:
    """LRU cache implementation"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
        
    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value

class EnhancedEvolutionCache:
    """Multi-level evolution prediction cache system"""
    def __init__(self, max_cache_size=1000):
        self.prototype_cache = LRUCache(max_cache_size // 2)
        self.pattern_cache = LRUCache(max_cache_size // 4)
        self.batch_cache = LRUCache(max_cache_size // 4)
        
        self.hit_count = 0
        self.miss_count = 0
        
    def get_cache_key(self, tensor):
        """Generate efficient cache key"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        return hash(tensor.detach().cpu().numpy().tobytes())
        
    def get_batch_signature(self, targets, morph0_prototypes):
        """Generate batch signature for batch-level caching"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        class_indices = sorted([t.item() for t in targets if t.item() in morph0_prototypes])
        return tuple(class_indices)
    
    def get_hit_rate(self):
        """Get cache hit rate"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def clear_cache(self):
        """Clear all caches"""
        self.prototype_cache = LRUCache(self.prototype_cache.capacity)
        self.pattern_cache = LRUCache(self.pattern_cache.capacity)
        self.batch_cache = LRUCache(self.batch_cache.capacity)
        self.hit_count = 0
        self.miss_count = 0

class MorphologyMemoryPool(nn.Module):
    """
    Evolution-aware Memory Pool for Stage-CIL
    [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
    """
    def __init__(self, feature_dim, num_classes_per_pair, pool_size=512, hidden_dim=1024):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes_per_pair = num_classes_per_pair
        self.pool_size = pool_size
        self.hidden_dim = hidden_dim
        
        # Core components - structure preserved
        self.memory_keys = nn.Parameter(torch.randn(pool_size, feature_dim))
        self.memory_values = nn.Parameter(torch.randn(pool_size, feature_dim))
        
        # Evolution network - architecture preserved, implementation hidden
        self.evolution_net = nn.Sequential(
            # [NETWORK ARCHITECTURE WILL BE RELEASED AFTER ACCEPTANCE]
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Attention mechanism - structure preserved
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        
        # Evolution patterns pool - size preserved
        self.evolution_patterns = nn.Parameter(torch.randn(50, feature_dim))
        device = self.evolution_patterns.device
        self.pattern_usage_count = torch.zeros(50, device=device)
        self.pattern_quality = torch.zeros(50, device=device)
        
        # Temperature parameter - value hidden
        self.softmax_temperature = 0.1  # [VALUE WILL BE RELEASED AFTER ACCEPTANCE]
        
        # Cache component - structure preserved
        self.evolution_cache = EnhancedEvolutionCache(max_cache_size=500)
        
        # Performance tracking - structure preserved
        self.performance_tracker = {
            'cache_hit_rate': 0.0,
            'compute_time': [],
            'pattern_usage': torch.zeros(50, device=self.evolution_patterns.device)
        }
        
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        nn.init.xavier_uniform_(self.memory_keys)
        nn.init.xavier_uniform_(self.memory_values)
        nn.init.xavier_uniform_(self.evolution_patterns)
        
        for m in self.evolution_net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, morph0_features, mode='evolve'):
        """
        Forward propagation - simplified version
        """
        if mode == 'evolve':
            return self.fast_predict_evolution(morph0_features)
        else:
            return self._update_memory_patterns(morph0_features)

    def fast_predict_evolution(self, morph0_features):
        """Fast evolution prediction using fixed pattern pool"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        start_time = time.time()
        batch_size = morph0_features.size(0)
        
        # Cache checking and prediction logic
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        
        return morph0_features, None

    def _predict_with_fixed_patterns(self, morph0_features):
        """Predict evolution using fixed pattern pool"""
        # [TOP-K SELECTION LOGIC WILL BE RELEASED AFTER ACCEPTANCE]
        # [ATTENTION MECHANISM IMPLEMENTATION WILL BE RELEASED AFTER ACCEPTANCE]
        # [EVOLUTION PREDICTION LOGIC WILL BE RELEASED AFTER ACCEPTANCE]
        batch_size = morph0_features.size(0)
        feature_dim = morph0_features.size(-1)
        
        # Pattern selection and attention aggregation
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        
        return morph0_features, None

    def update_evolution_patterns(self, morph0_features, morph1_features, learning_rate=0.01):
        """
        Update evolution patterns using competitive online learning
        [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        """
        # [COMPETITIVE UPDATE RULE WILL BE RELEASED AFTER ACCEPTANCE]
        # [ADAPTIVE TOP-K SELECTION WILL BE RELEASED AFTER ACCEPTANCE]
        # [WEIGHTED UPDATE MECHANISM WILL BE RELEASED AFTER ACCEPTANCE]
        with torch.no_grad():
            # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
            pass

    def compute_evolution_loss(self, morph0_features, morph1_features, predicted_morph1):
        """Compute evolution loss"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        if predicted_morph1 is None:
            return torch.tensor(0.0, device=morph1_features.device)
        
        # [LOSS COMPUTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        return torch.tensor(0.0, device=morph1_features.device)

    def smart_rehearsal_loss(self, current_features, target_classes):
        """Simplified evolution rehearsal loss"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        return torch.tensor(0.0, device=current_features.device)

    def compute_rehearsal_loss(self, current_features, target_classes):
        """Compute evolution rehearsal loss - maintain backward compatibility"""
        return self.smart_rehearsal_loss(current_features, target_classes)

    def get_memory_status(self):
        """Get memory pool status"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        status = {
            'pool_size': self.pool_size,
            'fixed_patterns': len(self.evolution_patterns),
            'cache_hit_rate': self.evolution_cache.get_hit_rate(),
            'avg_compute_time': 0.0,
            'pattern_usage_stats': {
                'max_usage': 0.0,
                'min_usage': 0.0,
                'avg_usage': 0.0
            },
            'pattern_usage_count': self.pattern_usage_count.clone().cpu().numpy(),
            'pattern_quality': self.pattern_quality.clone().cpu().numpy()
        }
        return status
    
    def report_optimization_status(self):
        """Report optimization status"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        status = self.get_memory_status()
        logging.info("=== Memory Pool Status Report ===")
        logging.info(f"  Fixed pattern pool size: {status['fixed_patterns']}")
        logging.info(f"  Cache hit rate: {status['cache_hit_rate']:.2f}%")

    def report_pattern_usage_statistics(self):
        """Report pattern usage statistics"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        logging.info("=== Evolution Pattern Usage Statistics ===")

    def output_all_pattern_usage(self):
        """Output usage frequency for all 50 patterns"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        logging.info("=== All 50 Evolution Pattern Usage Frequencies ===")

    def update_memory(self, morph0_features, morph1_features, learning_rate=0.01):
        """Online memory pool update using competitive update"""
        self.update_evolution_patterns(morph0_features, morph1_features, learning_rate)

class MorphologyEvolutionLearner(BaseLearner):
    """
    Morphology Evolution Incremental Learner
    """
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = Proof_Net(args, False)
        self._network.extend_task()
        self._network.update_context_prompt()
        
        # Morphology evolution related parameters
        self.batch_size = get_attribute(args, "batch_size", 48)
        self.init_lr = get_attribute(args, "init_lr", 0.01)
        self.weight_decay = get_attribute(args, "weight_decay", 0.0005)
        self.min_lr = get_attribute(args, "min_lr", 1e-8)
        self.tuned_epoch = get_attribute(args, "tuned_epoch", 5)
        
        # Morphology management
        self._known_classes = 0
        self._current_morphology_pair = -1
        self._morphology_stage = 0
        self._classes_per_pair = get_attribute(args, "classes_per_pair", 5)
        
        # Memory Pool configuration
        feature_dim = self._network.feature_dim
        self.memory_pool = MorphologyMemoryPool(
            feature_dim=feature_dim,
            num_classes_per_pair=self._classes_per_pair,
            pool_size=get_attribute(args, "memory_pool_size", 512),
            hidden_dim=get_attribute(args, "memory_pool_hidden", 1024)
        ).to(self._device)
        
        # Morphology evolution prototype storage
        self._morphology_prototypes = {}
        
        # Data collector for visualization
        self.data_collector = MorphologyDataCollector()
        
        # Loss weight configuration
        self.lambda_evolution = get_attribute(args, "lambda_evolution", 0.5)
        self.lambda_rehearsal = get_attribute(args, "lambda_rehearsal", 0.3)
        self.lambda_memory = get_attribute(args, "lambda_memory", 0.2)
        
        logging.info(f"Morphology evolution learner initialization completed")
        
        self._cached_text_features = None
        self._cached_text_classes = 0
        self._cached_text_templates = None
        
        # Task-level accuracy tracking
        self._task_specific_accuracies = {}

    def incremental_train(self, data_manager, task_idx=None, morphology_stage=None, pair_idx=None):
        """
        Morphology evolution incremental training
        
        Args:
            data_manager: Data manager
            task_idx: Task index
            morphology_stage: Morphology stage (0: Stage 0, 1: Stage 1)
            pair_idx: Morphology pair index
        """
        # Update current state
        if task_idx is not None:
            self._cur_task = task_idx
        if morphology_stage is not None:
            self._morphology_stage = morphology_stage
        if pair_idx is not None:
            self._current_morphology_pair = pair_idx
            
        logging.info(f"Starting morphology evolution training:")
        logging.info(f"  Task index: {self._cur_task}")
        logging.info(f"  Morphology stage: {self._morphology_stage}")
        logging.info(f"  Morphology pair index: {self._current_morphology_pair}")
        
        # Get actual number of classes in current task from DataManager
        num_classes_in_task = data_manager.get_task_size(self._cur_task)
        logging.info(f"Number of classes in current task from DataManager: {num_classes_in_task}")

        # Morphology evolution class count management
        if self._morphology_stage == 0:
            self._total_classes = self._known_classes + num_classes_in_task
        else:
            self._total_classes = self._known_classes + num_classes_in_task
        
        logging.info(f"  Known classes: {self._known_classes} -> {self._total_classes}")
        
        # Update network prototypes
        self._network.update_prototype(self._total_classes)
        
        # Get training data
        train_dataset = data_manager.get_multimodal_dataset(
            self._cur_task, source="train", mode="train", 
            appendent=self._get_memory()
        )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        
        # Save old network
        self._old_network = copy.deepcopy(self._network).to(self._device)
        self._old_network.eval()
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=num_workers
        )
        
        # Create test loaders
        self.current_test_loader, self.test_loader = self._create_test_loaders(data_manager)
        
        # Compute prototypes
        train_dataset_for_proto = data_manager.get_multimodal_dataset(
            self._cur_task, source="train", mode="test"
        )
        train_loader_for_proto = DataLoader(
            train_dataset_for_proto, batch_size=self.batch_size,
            shuffle=True, num_workers=num_workers
        )
        
        # GPU parallel processing
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
            self.memory_pool = nn.DataParallel(self.memory_pool, self._multiple_gpus)
            
        self._network.to(self._device)
        self.memory_pool.to(self._device)
        
        # Compute current task prototypes
        self._compute_morphology_prototypes(train_loader_for_proto)
        
        # Execute morphology evolution training
        if self._morphology_stage == 0:
            self._train_morphology_0()
        else:
            self._train_morphology_1()
            
        # Build rehearsal memory
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        
        # Restore single GPU mode
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
            
        # Update known_classes (only after Stage 1 completion)
        if self._morphology_stage == 1:
            self._known_classes = self._total_classes
        
        # Collect training data for visualization
        self._collect_training_data(data_manager)

        # Save final model weights after last task increment
        if (task_idx is not None and morphology_stage is not None and
            task_idx == data_manager.nb_tasks - 1 and morphology_stage == 1):
            save_path = 'stag_final_incremental_model.pth'
            torch.save(self._network.state_dict(), save_path)
            logging.info(f"Final incremental model saved to {save_path}")

    def _collect_training_data(self, data_manager):
        """Collect training data for visualization analysis"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        logging.info("Starting to collect training data for visualization analysis...")
        
        try:
            # Collect accuracy metrics, feature data, distance metrics
            # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
            logging.info(f"Task {self._cur_task} morphology {self._morphology_stage} data collection completed")
            
        except Exception as e:
            logging.error(f"Data collection failed: {e}")

    def _compute_morphology_prototypes(self, train_loader):
        """Compute morphology prototypes"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        self._network.eval()
        
        # Initialize prototype storage for current morphology pair
        if self._current_morphology_pair not in self._morphology_prototypes:
            self._morphology_prototypes[self._current_morphology_pair] = {}
            
        # [PROTOTYPE COMPUTATION LOGIC WILL BE RELEASED AFTER ACCEPTANCE]

    def _train_morphology_0(self):
        """Train Stage 0 (basic learning)"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        logging.info("Starting Stage 0 training (basic learning)")
        
        # Training setup
        self._setup_optimizer_and_scheduler()
        
        # Training loop
        for epoch in range(self.tuned_epoch):
            # [TRAINING LOOP IMPLEMENTATION WILL BE RELEASED AFTER ACCEPTANCE]
            pass

    def _train_morphology_1(self):
        """Train Stage 1 (evolution learning)"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        logging.info("Starting Stage 1 training (evolution learning)")
        
        # Get Stage 0 prototypes for evolution learning
        morph0_prototypes = self._get_morphology_0_prototypes()
        
        if not morph0_prototypes:
            logging.warning("No Stage 0 prototypes found, using standard training")
            self._train_morphology_0()
            return
            
        # Training setup
        self._setup_optimizer_and_scheduler()
        
        # Training loop
        for epoch in range(self.tuned_epoch):
            # [TRAINING LOOP IMPLEMENTATION WILL BE RELEASED AFTER ACCEPTANCE]
            pass

    def _get_morphology_0_prototypes(self):
        """Get Stage 0 prototypes"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        if self._current_morphology_pair not in self._morphology_prototypes:
            return {}
            
        morph0_prototypes = {}
        pair_prototypes = self._morphology_prototypes[self._current_morphology_pair]
        
        for class_idx, stages in pair_prototypes.items():
            if 0 in stages:
                morph0_prototypes[class_idx] = stages[0].to(self._device)
                
        return morph0_prototypes

    def _predict_evolution_from_prototypes(self, targets, morph0_prototypes):
        """Predict Stage 1 from Stage 0 prototypes using Memory Pool"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        batch_size = targets.size(0)
        device = targets.device
        feature_dim = 512
        
        # [EVOLUTION PREDICTION LOGIC WILL BE RELEASED AFTER ACCEPTANCE]
        
        return None, None

    def _compute_memory_pool_loss(self, morph1_features, targets):
        """Compute Memory Pool contrastive learning loss"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        return torch.tensor(0.0, device=self._device)

    def _update_memory_pool_online(self, targets, morph0_prototypes, morph1_features):
        """Intelligent online Memory Pool update"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        with torch.no_grad():
            # [ONLINE UPDATE LOGIC WILL BE RELEASED AFTER ACCEPTANCE]
            pass

    def _compute_morphology_rehearsal_loss(self):
        """Compute morphology evolution prototype rehearsal loss"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        return torch.tensor(0.0, device=self._device)

    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and scheduler"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        # Freeze backbone, only train necessary parts
        for name, param in self._network.convnet.named_parameters():
            if 'logit_scale' not in name:
                param.requires_grad = False
                
        self._network.freeze_projection_weight_new()
        
        # Collect parameters to optimize
        params_to_optimize = list(self._network.parameters()) + list(self.memory_pool.parameters())
        
        if self.args['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(
                params_to_optimize, momentum=0.9, 
                lr=self.init_lr, weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            self.optimizer = torch.optim.AdamW(
                params_to_optimize, lr=self.init_lr, 
                weight_decay=self.weight_decay
            )
        else:
            self.optimizer = torch.optim.AdamW(
                params_to_optimize, lr=self.init_lr, 
                weight_decay=self.weight_decay
            )
            
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.tuned_epoch, eta_min=self.min_lr
        )

    def _forward_for_classification(self, images, text_list=None):
        """Forward pass for classification"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        image_features = self._network.encode_image(images)
        image_features = F.normalize(image_features, dim=1)
        
        if text_list is None:
            text_features = self._get_cached_text_features(self._total_classes)
        else:
            # [TEXT ENCODING LOGIC WILL BE RELEASED AFTER ACCEPTANCE]
            text_features = torch.zeros(len(text_list), image_features.size(1)).to(self._device)
            
        logits = image_features @ text_features.t()
        return logits

    def _forward_for_classification_features(self, image_features, text_list):
        """Forward pass for classification using given features"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        image_features = F.normalize(image_features, dim=1)
        
        with torch.no_grad():
            texts_tokenized = self._network.tokenizer(text_list).to(self._device)
            text_features = self._network.encode_text(texts_tokenized)
            text_features = F.normalize(text_features, dim=1)
            
        logits = image_features @ text_features.t()
        return logits

    def _create_test_loaders(self, data_manager):
        """Create test data loaders"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        try:
            # Current task test set
            current_test_dataset = data_manager.get_multimodal_dataset(
                self._cur_task, source="test", mode="test"
            )
            current_test_loader = DataLoader(
                current_test_dataset, batch_size=self.batch_size,
                shuffle=False, num_workers=num_workers
            )
            
            # Cumulative test set
            test_datasets = []
            for task_idx in range(self._cur_task + 1):
                try:
                    task_test_dataset = data_manager.get_multimodal_dataset(
                        task_idx, source="test", mode="test"
                    )
                    if len(task_test_dataset) > 0:
                        test_datasets.append(task_test_dataset)
                except Exception as e:
                    logging.warning(f"Failed to get task {task_idx} test data: {e}")
                    continue
                    
            if test_datasets:
                from torch.utils.data import ConcatDataset
                cumulative_test_dataset = ConcatDataset(test_datasets)
                cumulative_test_loader = DataLoader(
                    cumulative_test_dataset, batch_size=self.batch_size,
                    shuffle=False, num_workers=num_workers
                )
            else:
                cumulative_test_loader = current_test_loader
                
            return current_test_loader, cumulative_test_loader
            
        except Exception as e:
            logging.error(f"Failed to create test loaders: {e}")
            return None, None

    def eval_morphology_evolution(self, data_manager, pair_idx):
        """Evaluate morphology evolution performance"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        if pair_idx not in self._morphology_prototypes:
            return 0.0
            
        return 0.0

    def report_memory_pool_status(self, pair_idx):
        """Report Memory Pool status"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        status = self.memory_pool.get_memory_status()
        
        logging.info(f"=== Memory Pool Status Report (Morphology Pair {pair_idx}) ===")
        logging.info(f"  Memory pool size: {status['pool_size']}")
        logging.info(f"  Fixed pattern pool size: {status['fixed_patterns']}")
        logging.info(f"  Cache hit rate: {status['cache_hit_rate']:.2%}")

    @torch.no_grad()
    def _compute_accuracy(self, model, loader):
        """Compute accuracy"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        model.eval()
        correct, total = 0, 0
        
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt[0]
        all_labels = class_to_label[:self._total_classes]
        
        for batch in loader:
            # [ACCURACY COMPUTATION LOGIC WILL BE RELEASED AFTER ACCEPTANCE]
            pass
            
        return np.around((correct / total) * 100, decimals=2)

    def _eval_cnn(self, loader):
        """Evaluate using Memory Pool enhanced prediction"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        self._network.eval()
        y_pred, y_true = [], []
        total_samples = 0
        
        logging.info(f"Starting test evaluation...")
        
        for batch in loader:
            # [EVALUATION LOGIC WILL BE RELEASED AFTER ACCEPTANCE]
            pass
            
        # Ensure correct return format
        if y_pred and y_true:
            y_pred_array = np.concatenate(y_pred, axis=0)
            y_true_array = np.concatenate(y_true, axis=0)
            
            # Calculate Top-1 accuracy
            top1_correct = (y_pred_array[:, 0] == y_true_array).sum()
            top1_accuracy = (top1_correct / total_samples) * 100
            
            logging.info(f"Test completed:")
            logging.info(f"  Test samples: {total_samples}")
            logging.info(f"  Top-1 accuracy: {top1_accuracy:.2f}%")
            
            return y_pred_array, y_true_array
        else:
            logging.info(f"Test completed: No valid samples")
            empty_pred = np.zeros((0, self.topk), dtype=np.int64)
            empty_true = np.zeros((0,), dtype=np.int64)
            return empty_pred, empty_true

    def eval_task(self):
        """Evaluate task performance"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        # Incremental learning specific evaluation
        incremental_metrics = self._compute_incremental_metrics()

        if self.args["convnet_type"].lower() != "clip" or self.args["model_name"].lower() == "l2p" or self.args["model_name"].lower() == "dualprompt":
            return cnn_accy, nme_accy, None, None, None, None, incremental_metrics
        else:
            y_pred, y_true = self._eval_zero_shot()
            zs_acc = self._evaluate_zs(y_pred, y_true)
            zs_seen, zs_unseen, zs_harmonic, zs_total = zs_acc["grouped"]["old"], zs_acc["grouped"]["new"], zs_acc["grouped"]["harmonic"], zs_acc["grouped"]["total"]

        return cnn_accy, nme_accy, zs_seen, zs_unseen, zs_harmonic, zs_total, incremental_metrics

    def _compute_incremental_metrics(self):
        """Compute incremental learning metrics"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        if not hasattr(self, '_task_accuracies'):
            self._task_accuracies = []
            
        # Calculate current task accuracy on all known classes
        current_acc = self._compute_accuracy_on_all_tasks()
        self._task_accuracies.append(current_acc)
        
        # Calculate various incremental learning metrics
        metrics = {
            'current_task_acc': current_acc,
            'average_incremental_acc': self._compute_average_incremental_accuracy(),
            'forgetting_rate': self._compute_forgetting_rate(),
            'backward_transfer': self._compute_backward_transfer(),
            'forward_transfer': self._compute_forward_transfer(),
            'all_task_accuracies': self._task_accuracies.copy()
        }
        
        return metrics

    def _compute_accuracy_on_all_tasks(self):
        """Compute cumulative accuracy on all learned tasks"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        return 0.0

    def _compute_average_incremental_accuracy(self):
        """Compute average incremental accuracy"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        if not hasattr(self, '_task_accuracies') or len(self._task_accuracies) == 0:
            return 0.0
        return sum(self._task_accuracies) / len(self._task_accuracies)

    def _compute_forgetting_rate(self):
        """Compute forgetting rate"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        return 0.0

    def _compute_task_specific_accuracy(self, task_idx):
        """Compute accuracy on specific task"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        return 0.0

    def _compute_backward_transfer(self):
        """Compute backward transfer (performance improvement on previous tasks)"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        return 0.0

    def _compute_forward_transfer(self):
        """Compute forward transfer (help for new tasks)"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        return 0.0

    def _compute_distillation_loss(self, inputs, targets):
        """Compute knowledge distillation loss"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        if self._old_network is None:
            return torch.tensor(0.0, device=self._device)
        
        try:
            # [DISTILLATION LOSS COMPUTATION WILL BE RELEASED AFTER ACCEPTANCE]
            return torch.tensor(0.0, device=self._device)
        except Exception as e:
            logging.debug(f"Knowledge distillation computation failed: {e}")
            return torch.tensor(0.0, device=self._device)

    def _get_memory(self):
        """Get historical samples for rehearsal"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        if self._cur_task == 0:
            return None
        
        # Check if stored memory data exists
        if not hasattr(self, '_data_memory') or self._data_memory is None:
            logging.info("No stored memory data, returning None")
            return None
        
        # [MEMORY RETRIEVAL LOGIC WILL BE RELEASED AFTER ACCEPTANCE]
        return None

    def build_rehearsal_memory(self, data_manager, per_class):
        """Build rehearsal memory"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        # Call parent method to update prototypes
        super().build_rehearsal_memory(data_manager, per_class)
        
        # Get current task data for storage
        try:
            current_data, current_targets, current_stages, _ = data_manager.get_dataset(
                cil_task_idx=self._cur_task,
                source="train",
                mode="test",
                ret_data=True
            )
            
            if len(current_data) == 0:
                logging.warning(f"Current task {self._cur_task} has no available data")
                return
            
            # [MEMORY BUILDING LOGIC WILL BE RELEASED AFTER ACCEPTANCE]
                
        except Exception as e:
            logging.error(f"Failed to build rehearsal memory: {e}")
    
    def _balance_memory_size(self):
        """Balance memory size"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        if not hasattr(self, '_data_memory') or len(self._data_memory) <= self._memory_size:
            return
            
        # [MEMORY BALANCING LOGIC WILL BE RELEASED AFTER ACCEPTANCE]

    def _get_cached_text_features(self, total_classes):
        """Get cached text features to avoid repeated encoding"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        if (self._cached_text_features is None or 
            self._cached_text_classes != total_classes or
            self._cached_text_templates != self.data_manager._data_to_prompt[0]):
            class_to_label = self.data_manager._class_to_label
            templates = self.data_manager._data_to_prompt[0]
            total_labels = class_to_label[:total_classes]
            text_batch = [templates.format(lbl) for lbl in total_labels]
            with torch.no_grad():
                texts_tokenized = self._network.tokenizer(text_batch).to(self._device)
                self._cached_text_features = self._network.encode_text(texts_tokenized)
                self._cached_text_features = F.normalize(self._cached_text_features, dim=1)
                self._cached_text_classes = total_classes
                self._cached_text_templates = templates
        return self._cached_text_features

    def update_evolution_history(self, class_idx, morph0_prototype, morph1_prototype):
        """Update evolution history"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        morph0_tensor = morph0_prototype.unsqueeze(0)
        morph1_tensor = morph1_prototype.unsqueeze(0)
        self.update_evolution_patterns(morph0_tensor, morph1_tensor, learning_rate=0.01)

    def _collect_accuracy_metrics(self, data_manager):
        """Collect accuracy metrics"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        metrics = {}
        return metrics
    
    def _collect_feature_data(self, data_manager):
        """Collect feature data"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        feature_data = {}
        return feature_data
    
    def _collect_distance_metrics(self):
        """Collect distance metrics"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        distance_metrics = {}
        return distance_metrics
    
    def _collect_evolution_analysis_data(self):
        """Collect evolution analysis data"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        try:
            # [EVOLUTION ANALYSIS LOGIC WILL BE RELEASED AFTER ACCEPTANCE]
            pass
        except Exception as e:
            logging.warning(f"Evolution analysis data collection failed: {e}")
    
    def _compute_stage_intra_accuracy(self, data_manager):
        """Compute intra-stage accuracy"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        return 0.0
    
    def _compute_stage_inter_accuracy(self, data_manager):
        """Compute inter-stage accuracy"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        return 0.0
    
    def _extract_features_and_labels(self, data_loader):
        """Extract features and labels from data loader"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        self._network.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            # [FEATURE EXTRACTION LOGIC WILL BE RELEASED AFTER ACCEPTANCE]
            pass
        
        if all_features:
            features_tensor = torch.cat(all_features, dim=0)
            labels_tensor = torch.cat(all_labels, dim=0)
            return features_tensor, labels_tensor
        else:
            return None, None
    
    def _get_class_features(self, class_idx):
        """Get features for specific class"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        return None
    
    def _compute_intra_class_distance(self, prototype, features):
        """Compute intra-class distance"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        return 0.0
    
    def _compute_morphology_inter_distance(self, morph0_prototypes, morph1_prototypes):
        """Compute morphology inter-distance"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        return 0.0
    
    def _get_morphology_features(self, morphology_stage):
        """Get morphology features"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        return None
    
    def save_final_visualization_data(self, save_path="visualization_data/morphology_evolution_data.pkl"):
        """Save final visualization data"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        self.data_collector.save_data(save_path)
    
    def eval_morphology_forgetting(self, data_manager, task_range=None, stage_specific=False):
        """Evaluate morphology forgetting"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        return 0.0, [], 0.0, [], {}, {}

    def _compute_class_specific_forgetting(self, data_manager, class_list):
        """Compute class-specific forgetting"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        return 0.0

    def _compute_class_stage_accuracy(self, data_manager, class_idx, stage):
        """Compute class stage accuracy"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        return 0.0

    def load_final_model(self, model_path='final_incremental_model.pth'):
        """Load final model"""
        # [IMPLEMENTATION DETAILS WILL BE RELEASED AFTER ACCEPTANCE]
        if os.path.exists(model_path):
            self._network.load_state_dict(torch.load(model_path))
            logging.info(f"Final model loaded from {model_path}")
        else:
            logging.warning(f"Model file {model_path} not found")


# For compatibility, create an alias
Learner = MorphologyEvolutionLearner
    

