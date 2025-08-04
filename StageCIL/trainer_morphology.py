import sys
import logging
import copy
import torch
import numpy as np
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import time
from datetime import datetime


def train(args):
    """Main training function for morphology evolution incremental learning"""
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train_morphology(args)


def _train_morphology(args):
    """
    Main morphology evolution training logic:
    - Each pair of tasks represents different stages of the same classes (Stage 0 → Stage 1)
    - Maintains Memory Pool to learn morphological evolution transformations
    """
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    
    # Create log directory
    logs_name = "logs/{}/{}/{}/{}".format(
        args["model_name"], args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    # Configure log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}_morphology_{}".format(
        args["model_name"], args["dataset"], 
        init_cls, args["increment"], args["prefix"], 
        args["seed"], args["convnet_type"], timestamp)
    
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)
    
    # Initialize data manager
    data_manager = DataManager(
        args["dataset"], args["shuffle"], args["seed"], 
        args["init_cls"], args["increment"]
    )
    
    # Log dataset information
    logging.info(f"=== Stage-Aware Class-Incremental Learning Setup ===")
    logging.info(f"Dataset: {args['dataset']}")
    logging.info(f"Total classes: {data_manager.get_total_classnum()}")
    logging.info(f"Initial classes: {args['init_cls']}")
    logging.info(f"Increment step: {args['increment']}")
    logging.info(f"Total tasks: {data_manager.nb_tasks}")
    
    # Calculate number of morphology pairs (every two tasks form a pair)
    num_pairs = data_manager.nb_tasks // 2
    logging.info(f"Morphology evolution pairs: {num_pairs} (each pair contains Stage 0→Stage 1)")
    
    # Create model
    model = factory.get_model(args["model_name"], args)
    model.save_dir = logs_name

    # Performance tracking metrics
    cnn_curve = {"top1": [], "top5": []}
    morphology_performance = []  # Track performance of each morphology pair
    
    # Train morphology pairs
    for pair_idx in range(num_pairs):
        logging.info(f"\n{'='*60}")
        logging.info(f"Starting morphology evolution pair {pair_idx+1}/{num_pairs}")
        logging.info(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Calculate corresponding task indices
        task0_idx = pair_idx * 2      # Stage 0 task index
        task1_idx = pair_idx * 2 + 1  # Stage 1 task index
        
        logging.info(f"Stage 0 task index: {task0_idx}, Stage 1 task index: {task1_idx}")
        
        # === Phase 1: Train Stage 0 ===
        logging.info(f"\n--- Phase 1: Train Stage 0 (Task {task0_idx}) ---")
        pair_start_time = time.time()
        
        # Train Stage 0
        stage0_start_time = time.time()
        model.incremental_train(data_manager, task_idx=task0_idx, morphology_stage=0, pair_idx=pair_idx)
        stage0_time = time.time() - stage0_start_time
        
        # Evaluate Stage 0
        eval_results_0 = model.eval_task()
        cnn_accy_0 = eval_results_0[0]
        incremental_metrics_0 = eval_results_0[-1] if len(eval_results_0) > 6 else None
        model.after_task()
        
        logging.info(f"Stage 0 training time: {stage0_time:.2f} seconds")
        logging.info(f"Stage 0 Top-1 accuracy: {cnn_accy_0['top1']:.2f}%")
        
        # Output incremental learning metrics
        if incremental_metrics_0:
            logging.info(f"=== Stage 0 Incremental Learning Metrics ===")
            logging.info(f"  Current task cumulative accuracy: {incremental_metrics_0['current_task_acc']:.2f}%")
            logging.info(f"  Average incremental accuracy: {incremental_metrics_0['average_incremental_acc']:.2f}%")
            logging.info(f"  Forgetting rate: {incremental_metrics_0['forgetting_rate']:.2f}%")
            logging.info(f"  Backward transfer: {incremental_metrics_0['backward_transfer']:.2f}%")
            if incremental_metrics_0['all_task_accuracies']:
                logging.info(f"  Task accuracy sequence: {[f'{acc:.2f}%' for acc in incremental_metrics_0['all_task_accuracies']]}")
        
        # === Phase 2: Train Stage 1 (using Memory Pool) ===
        logging.info(f"\n--- Phase 2: Train Stage 1 (Task {task1_idx}) ---")
        
        # Train Stage 1
        logging.info(f"\n{'='*50}")
        logging.info(f"Starting Stage 1 training - morphology pair {pair_idx}")
        logging.info(f"{'='*50}")
        
        # Report Memory Pool status before training
        model.report_memory_pool_status(pair_idx)
        
        stage1_start_time = time.time()
        model.incremental_train(
            data_manager, task_idx=task1_idx, 
            morphology_stage=1, pair_idx=pair_idx
        )
        stage1_time = time.time() - stage1_start_time
        
        # Evaluate Stage 1
        eval_results_1 = model.eval_task()
        cnn_accy_1 = eval_results_1[0]
        incremental_metrics_1 = eval_results_1[-1] if len(eval_results_1) > 6 else None
        model.after_task()
        
        logging.info(f"Stage 1 training time: {stage1_time:.2f} seconds")
        logging.info(f"Stage 1 Top-1 accuracy: {cnn_accy_1['top1']:.2f}%")
        
        # Output incremental learning metrics
        if incremental_metrics_1:
            logging.info(f"=== Stage 1 Incremental Learning Metrics ===")
            logging.info(f"  Current task cumulative accuracy: {incremental_metrics_1['current_task_acc']:.2f}%")
            logging.info(f"  Average incremental accuracy: {incremental_metrics_1['average_incremental_acc']:.2f}%")
            logging.info(f"  Forgetting rate: {incremental_metrics_1['forgetting_rate']:.2f}%")
            logging.info(f"  Backward transfer: {incremental_metrics_1['backward_transfer']:.2f}%")
            if incremental_metrics_1['all_task_accuracies']:
                logging.info(f"  Task accuracy sequence: {[f'{acc:.2f}%' for acc in incremental_metrics_1['all_task_accuracies']]}")
        
        # Record morphology pair performance (including incremental learning metrics)
        pair_performance = {
            'pair_idx': pair_idx,
            'morphology_0_acc': cnn_accy_0['top1'],
            'morphology_1_acc': cnn_accy_1['top1'],
            'avg_acc': (cnn_accy_0['top1'] + cnn_accy_1['top1']) / 2,
            # Incremental learning metrics
            'morph0_cumulative_acc': incremental_metrics_0['current_task_acc'] if incremental_metrics_0 else 0.0,
            'morph1_cumulative_acc': incremental_metrics_1['current_task_acc'] if incremental_metrics_1 else 0.0,
            'morph0_avg_inc_acc': incremental_metrics_0['average_incremental_acc'] if incremental_metrics_0 else 0.0,
            'morph1_avg_inc_acc': incremental_metrics_1['average_incremental_acc'] if incremental_metrics_1 else 0.0,
            'morph0_forgetting': incremental_metrics_0['forgetting_rate'] if incremental_metrics_0 else 0.0,
            'morph1_forgetting': incremental_metrics_1['forgetting_rate'] if incremental_metrics_1 else 0.0
        }
        morphology_performance.append(pair_performance)
        
        # Update overall performance curve
        cnn_curve["top1"].extend([cnn_accy_0['top1'], cnn_accy_1['top1']])
        if "top5" in cnn_accy_0 and "top5" in cnn_accy_1:
            cnn_curve["top5"].extend([cnn_accy_0['top5'], cnn_accy_1['top5']])
        
        # Calculate total time for current pair
        pair_total_time = time.time() - pair_start_time
        logging.info(f"Morphology pair {pair_idx+1} total time: {pair_total_time:.2f} seconds ({pair_total_time/60:.2f} minutes)")
        
        # Phase summary (including incremental learning metrics)
        logging.info(f"\n--- Morphology pair {pair_idx+1} completion summary ---")
        logging.info(f"Stage 0→Stage 1 accuracy change: {cnn_accy_0['top1']:.2f}% → {cnn_accy_1['top1']:.2f}%")
        
        if incremental_metrics_1:
            logging.info(f"=== Key Incremental Learning Metrics ===")
            logging.info(f"  [Core] Overall cumulative accuracy: {incremental_metrics_1['current_task_acc']:.2f}%")
            logging.info(f"  [Trend] Average incremental accuracy: {incremental_metrics_1['average_incremental_acc']:.2f}%")
            logging.info(f"  [Forgetting] Forgetting rate: {incremental_metrics_1['forgetting_rate']:.2f}%")
        
        logging.info(f"Morphology pair average accuracy: {pair_performance['avg_acc']:.2f}%")
        
        # Calculate cumulative average accuracy
        if len(cnn_curve["top1"]) > 0:
            cumulative_avg = sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
            logging.info(f"Cumulative average accuracy: {cumulative_avg:.2f}%")
    
    # === Final Results Summary ===
    logging.info(f"\n{'='*60}")
    logging.info("=== Stage-Aware Class-Incremental Learning Completed ===")
    
    # Overall performance statistics
    final_avg_acc = sum(cnn_curve["top1"]) / len(cnn_curve["top1"]) if cnn_curve["top1"] else 0.0
    logging.info(f"Final average accuracy: {final_avg_acc:.2f}%")
    logging.info(f"Complete accuracy curve: {[f'{acc:.2f}%' for acc in cnn_curve['top1']]}")
    
    # Morphology evolution performance summary (including incremental learning metrics)
    if morphology_performance:
        avg_morph0_acc = np.mean([p['morphology_0_acc'] for p in morphology_performance])
        avg_morph1_acc = np.mean([p['morphology_1_acc'] for p in morphology_performance])
        
        # Incremental learning metrics summary
        final_cumulative_acc = morphology_performance[-1]['morph1_cumulative_acc'] if morphology_performance else 0.0
        final_avg_inc_acc = morphology_performance[-1]['morph1_avg_inc_acc'] if morphology_performance else 0.0
        final_forgetting = morphology_performance[-1]['morph1_forgetting'] if morphology_performance else 0.0
        
        logging.info(f"\n=== [Final Results] Incremental Learning Performance Summary ===")
        logging.info(f"[Key Metric] Final overall cumulative accuracy: {final_cumulative_acc:.2f}%")
        logging.info(f"[Average Performance] Final average incremental accuracy: {final_avg_inc_acc:.2f}%")
        logging.info(f"[Forgetting Level] Final forgetting rate: {final_forgetting:.2f}%")
        
        logging.info(f"\n=== [Morphology Learning] Performance Summary ===")
        logging.info(f"Average Stage 0 accuracy: {avg_morph0_acc:.2f}%")
        logging.info(f"Average Stage 1 accuracy: {avg_morph1_acc:.2f}%")
        logging.info(f"Morphology evolution improvement: {(avg_morph1_acc - avg_morph0_acc):.2f}%")
    
    # Save detailed results
    results = {
        'cnn_curve': cnn_curve,
        'morphology_performance': morphology_performance,
        'final_avg_acc': final_avg_acc,
    }
    
    # Save detailed results
    results_path = os.path.join(logs_name, f"morphology_results_{timestamp}.npz")
    try:
        # Save results
        np.savez(
            results_path,
            **results  # Use unpacking operator to save all results
        )
        logging.info(f"Detailed results saved to {results_path}")
    except Exception as e:
        logging.error(f"Failed to save results: {e}")
        import traceback
        traceback.print_exc()
    
    logging.info("="*60)


def _set_device(args):
    """Set up device configuration"""
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def print_args(args):
    """Print all arguments for debugging"""
    for key, value in args.items():
        logging.info("{}: {}".format(key, value)) 