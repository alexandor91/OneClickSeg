import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import pickle
from typing import List, Dict, Tuple, Optional, Union
import random
from pathlib import Path
import h5py
import plyfile
from scipy.spatial import cKDTree

class ScanNet40Dataset(Dataset):
    """
    ScanNet40 Dataset for Interactive 3D Instance Segmentation
    """
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        num_clicks: Tuple[int, int] = (10, 30),
        max_points: int = 50000,
        transform=None,
        use_color: bool = True,
        use_normal: bool = True,
        cache_data: bool = True
    ):
        """
        Args:
            data_root: Path to ScanNet data root
            split: 'train', 'val', or 'test'
            num_clicks: (min, max) number of clicks to sample per instance
            max_points: Maximum number of points to sample from scene
            transform: Optional transform to apply
            use_color: Whether to include RGB colors
            use_normal: Whether to include surface normals
            cache_data: Whether to cache loaded data
        """
        self.data_root = Path(data_root)
        self.split = split
        self.num_clicks = num_clicks
        self.max_points = max_points
        self.transform = transform
        self.use_color = use_color
        self.use_normal = use_normal
        self.cache_data = cache_data
        
        # ScanNet40 semantic classes (0-39, with 40 being ignored)
        self.num_classes = 40
        self.ignore_label = 255
        
        # Load split files
        self.scenes = self._load_split_files()
        
        # Cache for loaded data
        self.data_cache = {} if cache_data else None
        
        print(f"Loaded {len(self.scenes)} scenes for {split} split")
    
    def _load_split_files(self) -> List[str]:
        """Load scene names for the given split"""
        split_file = self.data_root / f"scannetv2_{self.split}.txt"
        if not split_file.exists():
            # Fallback to manual split
            all_scenes = [d.name for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith('scene')]
            all_scenes.sort()
            
            # Simple split: 80% train, 10% val, 10% test
            n_scenes = len(all_scenes)
            if self.split == 'train':
                return all_scenes[:int(0.8 * n_scenes)]
            elif self.split == 'val':
                return all_scenes[int(0.8 * n_scenes):int(0.9 * n_scenes)]
            else:  # test
                return all_scenes[int(0.9 * n_scenes):]
        
        with open(split_file, 'r') as f:
            scenes = [line.strip() for line in f.readlines()]
        return scenes
    
    def _load_scene_data(self, scene_name: str) -> Dict:
        """Load point cloud and labels for a scene"""
        if self.cache_data and scene_name in self.data_cache:
            return self.data_cache[scene_name]
        
        scene_dir = self.data_root / scene_name
        
        # Load point cloud (.ply file)
        ply_path = scene_dir / f"{scene_name}_vh_clean_2.ply"
        if not ply_path.exists():
            ply_path = scene_dir / f"{scene_name}.ply"
        
        if not ply_path.exists():
            raise FileNotFoundError(f"PLY file not found for scene {scene_name}")
        
        # Read PLY file
        plydata = plyfile.PlyData.read(str(ply_path))
        vertices = plydata['vertex']
        
        # Extract coordinates
        coords = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1).astype(np.float32)
        
        # Extract features
        features = []
        if self.use_color:
            colors = np.stack([vertices['red'], vertices['green'], vertices['blue']], axis=1).astype(np.float32) / 255.0
            features.append(colors)
        
        if self.use_normal and all(k in vertices.dtype.names for k in ['nx', 'ny', 'nz']):
            normals = np.stack([vertices['nx'], vertices['ny'], vertices['nz']], axis=1).astype(np.float32)
            features.append(normals)
        
        if features:
            features = np.concatenate(features, axis=1)
        else:
            features = np.zeros((coords.shape[0], 0), dtype=np.float32)
        
        # Load semantic labels
        label_path = scene_dir / f"{scene_name}_vh_clean_2.labels.ply"
        if not label_path.exists():
            label_path = scene_dir / f"{scene_name}.labels.ply"
        
        semantic_labels = np.zeros(coords.shape[0], dtype=np.int32)
        if label_path.exists():
            label_plydata = plyfile.PlyData.read(str(label_path))
            if 'label' in label_plydata['vertex'].dtype.names:
                semantic_labels = label_plydata['vertex']['label'].astype(np.int32)
        
        # Load instance labels
        instance_path = scene_dir / f"{scene_name}_vh_clean_2.segs.json"
        if not instance_path.exists():
            instance_path = scene_dir / f"{scene_name}.segs.json"
        
        instance_labels = np.zeros(coords.shape[0], dtype=np.int32)
        if instance_path.exists():
            with open(instance_path, 'r') as f:
                seg_data = json.load(f)
            seg_indices = np.array(seg_data['segIndices'])
            instance_labels = seg_indices
        
        data = {
            'coords': coords,
            'features': features,
            'semantic_labels': semantic_labels,
            'instance_labels': instance_labels,
            'scene_name': scene_name
        }
        
        if self.cache_data:
            self.data_cache[scene_name] = data
        
        return data
    
    def _sample_clicks_from_instance(self, coords: np.ndarray, instance_mask: np.ndarray, 
                                   num_clicks: int) -> np.ndarray:
        """Sample click points from an instance"""
        instance_points = coords[instance_mask]
        if len(instance_points) == 0:
            return np.empty((0, 3), dtype=np.float32)
        
        # Sample random points from the instance
        num_clicks = min(num_clicks, len(instance_points))
        indices = np.random.choice(len(instance_points), num_clicks, replace=False)
        clicks = instance_points[indices]
        
        # Add small random noise to simulate user clicks
        noise = np.random.normal(0, 0.01, clicks.shape)
        clicks += noise
        
        return clicks.astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.scenes)
    
    def __getitem__(self, idx: int) -> Dict:
        scene_name = self.scenes[idx]
        data = self._load_scene_data(scene_name)
        
        coords = data['coords']
        features = data['features']
        semantic_labels = data['semantic_labels']
        instance_labels = data['instance_labels']
        
        # Subsample points if necessary
        if len(coords) > self.max_points:
            indices = np.random.choice(len(coords), self.max_points, replace=False)
            coords = coords[indices]
            features = features[indices]
            semantic_labels = semantic_labels[indices]
            instance_labels = instance_labels[indices]
        
        # Combine coordinates and features
        if features.shape[1] > 0:
            point_cloud = np.concatenate([coords, features], axis=1)
        else:
            point_cloud = coords
        
        # Get unique instances (excluding background)
        unique_instances = np.unique(instance_labels)
        unique_instances = unique_instances[unique_instances > 0]  # Remove background
        
        # Sample clicks for each instance
        clicks_per_instance = []
        masks_per_instance = []
        classes_per_instance = []
        
        for inst_id in unique_instances:
            instance_mask = (instance_labels == inst_id)
            
            # Skip very small instances
            if np.sum(instance_mask) < 10:
                continue
            
            # Get semantic class for this instance (majority vote)
            instance_semantic = semantic_labels[instance_mask]
            semantic_class = np.bincount(instance_semantic[instance_semantic != self.ignore_label]).argmax()
            
            # Sample clicks
            num_clicks = np.random.randint(self.num_clicks[0], self.num_clicks[1] + 1)
            clicks = self._sample_clicks_from_instance(coords, instance_mask, num_clicks)
            
            if len(clicks) > 0:
                clicks_per_instance.append(clicks)
                masks_per_instance.append(instance_mask)
                classes_per_instance.append(semantic_class)
        
        # Convert to tensors
        point_cloud = torch.from_numpy(point_cloud).float()
        
        # Format clicks as list of tensors
        click_tensors = [torch.from_numpy(clicks).float() for clicks in clicks_per_instance]
        
        # Create mask and class tensors
        if masks_per_instance:
            masks = torch.from_numpy(np.stack(masks_per_instance, axis=0)).float()
            classes = torch.from_numpy(np.array(classes_per_instance)).long()
        else:
            # Handle case with no valid instances
            masks = torch.zeros((0, len(point_cloud)), dtype=torch.float32)
            classes = torch.zeros(0, dtype=torch.long)
            click_tensors = []
        
        sample = {
            'point_cloud': point_cloud,
            'clicks': click_tensors,
            'masks': masks,
            'classes': classes,
            'scene_name': scene_name,
            'coords': torch.from_numpy(coords).float()  # Keep original coordinates
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class KITTI360Dataset(Dataset):
    """
    KITTI-360 Dataset for Interactive 3D Instance Segmentation
    """
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        sequences: Optional[List[str]] = None,
        num_clicks: Tuple[int, int] = (10, 30),
        max_points: int = 50000,
        transform=None,
        use_color: bool = True,
        use_intensity: bool = True,
        cache_data: bool = True
    ):
        """
        Args:
            data_root: Path to KITTI-360 data root
            split: 'train', 'val', or 'test'
            sequences: List of sequence IDs to use (e.g., ['2013_05_28_drive_0000_sync'])
            num_clicks: (min, max) number of clicks to sample per instance
            max_points: Maximum number of points to sample from scene
            transform: Optional transform to apply
            use_color: Whether to include RGB colors
            use_intensity: Whether to include laser intensity
            cache_data: Whether to cache loaded data
        """
        self.data_root = Path(data_root)
        self.split = split
        self.sequences = sequences
        self.num_clicks = num_clicks
        self.max_points = max_points
        self.transform = transform
        self.use_color = use_color
        self.use_intensity = use_intensity
        self.cache_data = cache_data
        
        # KITTI-360 has 19 semantic classes
        self.num_classes = 19
        self.ignore_label = 255
        
        # Load sequence data
        self.samples = self._load_samples()
        
        # Cache for loaded data
        self.data_cache = {} if cache_data else None
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self) -> List[Dict]:
        """Load all samples for the given split"""
        samples = []
        
        # Define default sequences if not provided
        if self.sequences is None:
            all_sequences = ['2013_05_28_drive_0000_sync', '2013_05_28_drive_0002_sync', 
                           '2013_05_28_drive_0003_sync', '2013_05_28_drive_0004_sync',
                           '2013_05_28_drive_0005_sync', '2013_05_28_drive_0006_sync',
                           '2013_05_28_drive_0007_sync', '2013_05_28_drive_0009_sync',
                           '2013_05_28_drive_0010_sync']
            
            # Simple split
            if self.split == 'train':
                self.sequences = all_sequences[:6]
            elif self.split == 'val':
                self.sequences = all_sequences[6:8]
            else:  # test
                self.sequences = all_sequences[8:]
        
        # Load samples from each sequence
        for seq in self.sequences:
            seq_path = self.data_root / 'data_3d_raw' / seq / 'velodyne_points' / 'data'
            if not seq_path.exists():
                continue
            
            # Get all .bin files in the sequence
            bin_files = sorted(seq_path.glob('*.bin'))
            for bin_file in bin_files:
                frame_id = bin_file.stem
                samples.append({
                    'sequence': seq,
                    'frame_id': frame_id,
                    'velodyne_path': bin_file
                })
        
        return samples
    
    def _load_velodyne_points(self, velodyne_path: Path) -> np.ndarray:
        """Load point cloud from velodyne .bin file"""
        points = np.fromfile(str(velodyne_path), dtype=np.float32).reshape(-1, 4)
        # points: [N, 4] where columns are [x, y, z, intensity]
        return points
    
    def _load_semantic_labels(self, sequence: str, frame_id: str) -> Optional[np.ndarray]:
        """Load semantic labels if available"""
        # KITTI-360 semantic labels are typically in .label format
        label_path = self.data_root / 'data_3d_semantics' / 'train' / sequence / 'labels' / f'{frame_id}.label'
        
        if not label_path.exists():
            return None
        
        labels = np.fromfile(str(label_path), dtype=np.uint32)
        semantic_labels = labels & 0xFFFF  # Lower 16 bits are semantic labels
        instance_labels = labels >> 16     # Upper 16 bits are instance labels
        
        return semantic_labels.astype(np.int32), instance_labels.astype(np.int32)
    
    def _sample_clicks_from_instance(self, coords: np.ndarray, instance_mask: np.ndarray, 
                                   num_clicks: int) -> np.ndarray:
        """Sample click points from an instance"""
        instance_points = coords[instance_mask]
        if len(instance_points) == 0:
            return np.empty((0, 3), dtype=np.float32)
        
        # Sample random points from the instance
        num_clicks = min(num_clicks, len(instance_points))
        indices = np.random.choice(len(instance_points), num_clicks, replace=False)
        clicks = instance_points[indices]
        
        # Add small random noise to simulate user clicks
        noise = np.random.normal(0, 0.02, clicks.shape)  # Slightly larger noise for outdoor scenes
        clicks += noise
        
        return clicks.astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample_info = self.samples[idx]
        sequence = sample_info['sequence']
        frame_id = sample_info['frame_id']
        velodyne_path = sample_info['velodyne_path']
        
        # Create cache key
        cache_key = f"{sequence}_{frame_id}"
        
        if self.cache_data and cache_key in self.data_cache:
            data = self.data_cache[cache_key]
        else:
            # Load velodyne points
            velodyne_points = self._load_velodyne_points(velodyne_path)
            coords = velodyne_points[:, :3]  # [x, y, z]
            intensity = velodyne_points[:, 3:4]  # intensity
            
            # Load labels if available
            label_data = self._load_semantic_labels(sequence, frame_id)
            if label_data is not None:
                semantic_labels, instance_labels = label_data
            else:
                # Create dummy labels for testing
                semantic_labels = np.zeros(len(coords), dtype=np.int32)
                instance_labels = np.zeros(len(coords), dtype=np.int32)
            
            data = {
                'coords': coords,
                'intensity': intensity,
                'semantic_labels': semantic_labels,
                'instance_labels': instance_labels,
                'sequence': sequence,
                'frame_id': frame_id
            }
            
            if self.cache_data:
                self.data_cache[cache_key] = data
        
        coords = data['coords']
        intensity = data['intensity']
        semantic_labels = data['semantic_labels']
        instance_labels = data['instance_labels']
        
        # Subsample points if necessary
        if len(coords) > self.max_points:
            indices = np.random.choice(len(coords), self.max_points, replace=False)
            coords = coords[indices]
            intensity = intensity[indices]
            semantic_labels = semantic_labels[indices]
            instance_labels = instance_labels[indices]
        
        # Build features
        features = []
        if self.use_intensity:
            features.append(intensity)
        
        # For KITTI-360, we typically don't have RGB colors from the lidar
        # But we can create pseudo-colors based on height or intensity
        if self.use_color:
            # Create height-based coloring
            z_normalized = (coords[:, 2] - coords[:, 2].min()) / (coords[:, 2].max() - coords[:, 2].min() + 1e-6)
            height_colors = np.stack([z_normalized, 1 - z_normalized, intensity.flatten()], axis=1)
            features.append(height_colors)
        
        if features:
            features = np.concatenate(features, axis=1)
            point_cloud = np.concatenate([coords, features], axis=1)
        else:
            point_cloud = coords
        
        # Get unique instances (excluding background)
        unique_instances = np.unique(instance_labels)
        unique_instances = unique_instances[unique_instances > 0]  # Remove background
        
        # Sample clicks for each instance
        clicks_per_instance = []
        masks_per_instance = []
        classes_per_instance = []
        
        for inst_id in unique_instances:
            instance_mask = (instance_labels == inst_id)
            
            # Skip very small instances
            if np.sum(instance_mask) < 20:  # Larger threshold for outdoor scenes
                continue
            
            # Get semantic class for this instance (majority vote)
            instance_semantic = semantic_labels[instance_mask]
            valid_semantics = instance_semantic[instance_semantic != self.ignore_label]
            if len(valid_semantics) == 0:
                continue
            
            semantic_class = np.bincount(valid_semantics).argmax()
            
            # Sample clicks
            num_clicks = np.random.randint(self.num_clicks[0], self.num_clicks[1] + 1)
            clicks = self._sample_clicks_from_instance(coords, instance_mask, num_clicks)
            
            if len(clicks) > 0:
                clicks_per_instance.append(clicks)
                masks_per_instance.append(instance_mask)
                classes_per_instance.append(semantic_class)
        
        # Convert to tensors
        point_cloud = torch.from_numpy(point_cloud).float()
        
        # Format clicks as list of tensors
        click_tensors = [torch.from_numpy(clicks).float() for clicks in clicks_per_instance]
        
        # Create mask and class tensors
        if masks_per_instance:
            masks = torch.from_numpy(np.stack(masks_per_instance, axis=0)).float()
            classes = torch.from_numpy(np.array(classes_per_instance)).long()
        else:
            # Handle case with no valid instances
            masks = torch.zeros((0, len(point_cloud)), dtype=torch.float32)
            classes = torch.zeros(0, dtype=torch.long)
            click_tensors = []
        
        sample = {
            'point_cloud': point_cloud,
            'clicks': click_tensors,
            'masks': masks,
            'classes': classes,
            'scene_name': f"{sequence}_{frame_id}",
            'coords': torch.from_numpy(coords).float(),
            'sequence': sequence,
            'frame_id': frame_id
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function to handle variable-length clicks
    """
    # Batch point clouds and coordinates
    point_clouds = [item['point_cloud'] for item in batch]
    coords_list = [item['coords'] for item in batch]
    scene_names = [item['scene_name'] for item in batch]
    
    # Handle clicks - they are already lists of tensors per scene
    all_clicks = []
    all_masks = []
    all_classes = []
    
    for item in batch:
        all_clicks.extend(item['clicks'])  # Flatten across batch
        if len(item['masks']) > 0:
            all_masks.append(item['masks'])
            all_classes.append(item['classes'])
    
    # Stack masks and classes if any exist
    if all_masks:
        masks = torch.cat(all_masks, dim=0)
        classes = torch.cat(all_classes, dim=0)
    else:
        masks = torch.zeros((0, point_clouds[0].size(0)), dtype=torch.float32)
        classes = torch.zeros(0, dtype=torch.long)
    
    return {
        'point_clouds': point_clouds,  # List of point clouds (different sizes)
        'coords_list': coords_list,    # List of coordinate tensors
        'clicks': all_clicks,          # List of click tensors
        'masks': masks,                # [total_instances, max_points] 
        'classes': classes,            # [total_instances]
        'scene_names': scene_names
    }


class RandomRotation:
    """Random rotation augmentation"""
    def __init__(self, axis: str = 'z', angle_range: Tuple[float, float] = (-np.pi, np.pi)):
        self.axis = axis
        self.angle_range = angle_range
    
    def __call__(self, sample: Dict) -> Dict:
        angle = np.random.uniform(*self.angle_range)
        
        if self.axis == 'z':
            # Rotation around z-axis (most common for indoor scenes)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            R = np.array([[cos_a, -sin_a, 0],
                         [sin_a, cos_a, 0],
                         [0, 0, 1]], dtype=np.float32)
        else:
            raise NotImplementedError(f"Rotation around {self.axis} axis not implemented")
        
        # Apply rotation to coordinates
        coords = sample['coords'].numpy()
        coords = coords @ R.T
        sample['coords'] = torch.from_numpy(coords).float()
        
        # Update point cloud coordinates
        point_cloud = sample['point_cloud'].numpy()
        point_cloud[:, :3] = coords
        sample['point_cloud'] = torch.from_numpy(point_cloud).float()
        
        # Apply rotation to clicks
        rotated_clicks = []
        for clicks in sample['clicks']:
            if len(clicks) > 0:
                clicks_np = clicks.numpy() @ R.T
                rotated_clicks.append(torch.from_numpy(clicks_np).float())
            else:
                rotated_clicks.append(clicks)
        sample['clicks'] = rotated_clicks
        
        return sample


class RandomScaling:
    """Random scaling augmentation"""
    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2)):
        self.scale_range = scale_range
    
    def __call__(self, sample: Dict) -> Dict:
        scale = np.random.uniform(*self.scale_range)
        
        # Apply scaling to coordinates
        coords = sample['coords'] * scale
        sample['coords'] = coords
        
        # Update point cloud coordinates
        point_cloud = sample['point_cloud'].clone()
        point_cloud[:, :3] = coords
        sample['point_cloud'] = point_cloud
        
        # Apply scaling to clicks
        scaled_clicks = []
        for clicks in sample['clicks']:
            if len(clicks) > 0:
                scaled_clicks.append(clicks * scale)
            else:
                scaled_clicks.append(clicks)
        sample['clicks'] = scaled_clicks
        
        return sample


# Example usage and testing
def create_dataloaders(
    scannet_root: Optional[str] = None,
    kitti_root: Optional[str] = None,
    batch_size: int = 4,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for both datasets
    """
    dataloaders = {}
    
    # Transforms
    train_transform = torch.nn.Sequential(
        RandomRotation(axis='z', angle_range=(-np.pi/4, np.pi/4)),
        RandomScaling(scale_range=(0.9, 1.1))
    )
    
    # ScanNet40 dataloaders
    if scannet_root and os.path.exists(scannet_root):
        for split in ['train', 'val']:
            dataset = ScanNet40Dataset(
                data_root=scannet_root,
                split=split,
                num_clicks=(10, 30),
                max_points=50000,
                transform=train_transform if split == 'train' else None,
                use_color=True,
                use_normal=True,
                cache_data=True
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True
            )
            
            dataloaders[f'scannet_{split}'] = dataloader
    
    # KITTI-360 dataloaders
    if kitti_root and os.path.exists(kitti_root):
        for split in ['train', 'val']:
            dataset = KITTI360Dataset(
                data_root=kitti_root,
                split=split,
                num_clicks=(15, 35),  # Slightly more clicks for outdoor scenes
                max_points=100000,    # More points for outdoor scenes
                transform=train_transform if split == 'train' else None,
                use_color=True,
                use_intensity=True,
                cache_data=True
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size // 2,  # Smaller batch size for larger scenes
                shuffle=(split == 'train'),
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True
            )
            
            dataloaders[f'kitti_{split}'] = dataloader
    
    return dataloaders


def test_dataloaders():
    """Test the dataloaders with dummy data"""
    # You would replace these with actual data paths
    scannet_root = "/path/to/scannet/data"
    kitti_root = "/path/to/kitti360/data"
    
    dataloaders = create_dataloaders(
        scannet_root=scannet_root if os.path.exists(scannet_root) else None,
        kitti_root=kitti_root if os.path.exists(kitti_root) else None,
        batch_size=2,
        num_workers=0
    )
    
    for name, dataloader in dataloaders.items():
        print(f"\nTesting {name} dataloader:")
        try:
            batch = next(iter(dataloader))
            print(f"  Point clouds: {len(batch['point_clouds'])} scenes")
            if len(batch['point_clouds']) > 0:
                print(f"  First scene shape: {batch['point_clouds'][0].shape}")
            print(f"  Total clicks: {len(batch['clicks'])}")
            print(f"  Total masks: {batch['masks'].shape}")
            print(f"  Total classes: {batch['classes'].shape}")
            print(f"  Scene names: {batch['scene_names']}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    test_dataloaders()