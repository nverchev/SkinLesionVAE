#@title Creating the dataset 

class SkinLesionsDataset(Dataset):
    def __init__(self, csv_file, root_dir, diagnose_list=None,
                     val_ratio = .2,test_ratio = .2,seed=1337):
        np.random.seed(seed)
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.diagnose_list=diagnose_list        
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.data=self._read_data()
        self.diagnose_id=self._diagnose_id()
        self.subset_indices=self._split_indices()

    def _read_data(self):
        data=pd.read_csv(self.csv_file)
        if self.diagnose_list:
          data = data.loc[data['dx'].isin(self.diagnose_list)].reset_index()               
        return data
    def _diagnose_id(self):
      return { string: i for i,string in enumerate(sorted(self.data['dx'].unique())) }
  
    def _split_indices(self):
        subset_indices={}
        dataset_size = len(self.data)
        indices = np.arange(dataset_size)
        val_dim = int(np.floor(self.val_ratio * dataset_size))
        test_dim = int(np.floor(self.test_ratio * dataset_size))
        train_dim = dataset_size-test_dim-val_dim
        subset_indices['all'] = np.random.shuffle(indices)
        subset_indices['train'] = indices[:train_dim]
        subset_indices['val'] = indices[train_dim:train_dim+val_dim]
        subset_indices['trainval'] = indices[:train_dim+val_dim]
        subset_indices['test'] = indices[train_dim+val_dim:]
        return subset_indices     
    def get_data(self,subset):
      try:
        indices=self.subset_indices[subset]
      except KeyError:
        raise ValueError('Invalid subset name.')
      subset_data=self.data.loc[indices].reset_index()

      image_names=[os.path.join(self.root_dir,
                                name+'.jpg') for name in subset_data['image_id']]
      images = [io.imread(img_name) for img_name in image_names]
 
      labels = [self.diagnose_id[diagnose] for diagnose in subset_data['dx']]
      return {'labels':labels, 'images':images}
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data['image_id'][idx])
        image = io.imread(img_name+'.jpg')
        label = self.diagnose_id[self.data['dx'][idx]]
        sample = [image, label]
        return sample

class SkinLesionsSubset(Dataset):
    def __init__(self, dataset, subset, transform=None):
      self.transform=transform
      self.data=dataset.get_data(subset)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.data['labels'][idx]
        image = self.data['images'][idx]
        if self.transform:
            image = self.transform(image)
        sample = [image, label]
        return sample

    def __len__(self):
        return len(self.data['labels'])

dset=SkinLesionsDataset('HAM10000_metadata.csv','HAM10000_images',diagnose_list=['nv','mel'])

