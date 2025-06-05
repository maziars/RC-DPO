from datasets import load_dataset, DatasetDict

def load_train_eval_datasets(train_dataset_name, eval_dataset_name=None, eval_ratio=0.05, seed=42, cache_path="cached_split", rank=0, world_size=1):
    if eval_dataset_name is None:
        dataset = load_dataset(train_dataset_name, split="train")
        if rank == 0:
            dataset = dataset.train_test_split(test_size=eval_ratio, seed=seed)
            dataset.save_to_disk(cache_path)
        import torch.distributed as dist
        dist.barrier()
        dataset = DatasetDict.load_from_disk(cache_path)
        train_dataset = dataset["train"].shard(num_shards=world_size, index=rank)
        eval_dataset = dataset["test"].shard(num_shards=world_size, index=rank)
    else:
        train_dataset = load_dataset(train_dataset_name, split="train").shard(num_shards=world_size, index=rank)
        eval_dataset = load_dataset(eval_dataset_name, split="train").shard(num_shards=world_size, index=rank)
    
    return train_dataset, eval_dataset