import os
in_path = '/NEW_EDS/JJ_Group/xutd/PixArt-alpha/bins_share/sam_sd_vae'
out_path = 'sam_512_sd_list.txt'

with open(out_path, "a+") as f:
    cnt = 0
    for root, dirs, files in os.walk(in_path):
        for file in files:
            line = os.path.join(root, file)
            f.write(line + '\n')
            cnt += 1
            if cnt % 1000 == 0:
                print("discover {} files".format(cnt))
