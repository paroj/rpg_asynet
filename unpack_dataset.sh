#!/bin/sh
# unpack dataset from within docker (requires unzip)
cd /media/user/Vault_4/HiWi_IGD/KITTI_dataset
#cd /opt/spool/vmechler/KITTI_Vision
echo $(ls)
apt-get install -y unzip
unzip data_object_label_2.zip
rm -f data_object_label_2.zip
unzip data_object_image_2.zip
rm -f data_object_image_2.zip
unzip data_object_prev_2.zip
rm -f data_object_prev_2.zip
echo DONE
echo $(ls)
cd training
echo $(ls)
echo finished
