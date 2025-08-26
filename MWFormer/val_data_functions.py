import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np

# --- Validation/test dataset --- #
class ValData(data.Dataset):
    def __init__(self, val_data_dir,val_filename, hybrid=False):
        super().__init__()
        val_list = val_data_dir + val_filename
        with open(val_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [i.strip().replace('input','gt') for i in input_names]  
            #gt_names = [i.strip().replace('Snow','Gt') for i in input_names]  #if using CSD dataset, then replace('Snow','Gt')
            if hybrid==False:
                gt_names = [i.strip().replace('rain','clean') for i in gt_names]   

        self.input_names = input_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        input_img = Image.open(self.val_data_dir + input_name)
        gt_img = Image.open(self.val_data_dir + gt_name)

        # Resizing image in the multiple of 16"
        wd_new,ht_new = input_img.size
        if ht_new>wd_new and ht_new>1024:
            wd_new = int(np.ceil(wd_new*1024/ht_new))
            ht_new = 1024
        elif ht_new<=wd_new and wd_new>1024:
            ht_new = int(np.ceil(ht_new*1024/wd_new))
            wd_new = 1024
        wd_new = int(16*np.ceil(wd_new/16.0))
        ht_new = int(16*np.ceil(ht_new/16.0))
        input_img = input_img.resize((wd_new,ht_new), Image.ANTIALIAS)
        gt_img = gt_img.resize((wd_new, ht_new), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    # normalize to [-1,1]
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_img)
        gt = transform_gt(gt_img)

        return input_im, gt, input_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)


#for images without gt
class ValData_unpaired(data.Dataset):
    def __init__(self, val_data_dir,val_filename):
        super().__init__()
        val_list = val_data_dir + val_filename
        with open(val_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]

        self.input_names = input_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        input_name = self.input_names[index]
        #print(input_name)
        input_img = Image.open(self.val_data_dir + input_name)

        # Resizing image in the multiple of 16"
        wd_new,ht_new = input_img.size
        if ht_new>wd_new and ht_new>1024:
            wd_new = int(np.ceil(wd_new*1024/ht_new))
            ht_new = 1024
        elif ht_new<=wd_new and wd_new>1024:
            ht_new = int(np.ceil(ht_new*1024/wd_new))
            wd_new = 1024
        wd_new = int(16*np.ceil(wd_new/16.0))
        ht_new = int(16*np.ceil(ht_new/16.0))
        input_img = input_img.resize((wd_new,ht_new), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    # normalize to [-1,1]
        input_im = transform_input(input_img)

        return input_im, input_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)


# # ================================================ WeatherStream dataset
class ValData_ws(data.Dataset):
    def __init__(self, val_data_dir,val_filename, hybrid=False):
        super().__init__()
        val_list = val_data_dir + val_filename
        with open(val_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            #input_names = [i.strip() for i in contents if "200" in i]   #to save some images

        self.input_names = input_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        input_name = self.input_names[index]

        #get gt name
        input_name_replaced = input_name.replace(".", "/")
        splitted_string = input_name_replaced.split("/")
        splitted_string[-2] = "gt"
        joined_string = "/".join(splitted_string)
        last_slash_index = joined_string.rfind("/")
        replaced_string = joined_string[:last_slash_index] + "." + joined_string[last_slash_index + 1:]
        
        input_img = Image.open(self.val_data_dir + input_name)
        gt_img = Image.open(self.val_data_dir + replaced_string)

        # Resizing image in the multiple of 16"
        wd_new,ht_new = input_img.size
        if ht_new>wd_new and ht_new>1024:
            wd_new = int(np.ceil(wd_new*1024/ht_new))
            ht_new = 1024
        elif ht_new<=wd_new and wd_new>1024:
            ht_new = int(np.ceil(ht_new*1024/wd_new))
            wd_new = 1024
        wd_new = int(16*np.ceil(wd_new/16.0))
        ht_new = int(16*np.ceil(ht_new/16.0))
        input_img = input_img.resize((wd_new,ht_new), Image.ANTIALIAS)
        gt_img = gt_img.resize((wd_new, ht_new), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    # normalize to [-1,1]
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_img)
        gt = transform_gt(gt_img)

        return input_im, gt, input_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)