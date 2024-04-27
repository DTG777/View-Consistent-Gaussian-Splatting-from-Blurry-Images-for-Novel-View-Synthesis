###########MOTION BLUR
# ##blurycozyroom
# python3 train.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
    # --train_ps 256 --train_dir datamotion/datacozyroom \
    # --val_ps 256 --val_dir datamotion/datacozyroom --env _0706 \
    # --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro --warmup  \
    # --source_path datamotion/datacozyroom --model_path datamotion/datacozyroom/output1000l1  --densify_until_iter 15000 --eval\
    # --resume  --pretrain_weights weights/blurycozyroom/model_best.pth --save_dir ./logs/blurycozyroom/3000
# #bluryfactory
# python3 train.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datamotion/datafactory \
#     --val_ps 256 --val_dir datafactory --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro --warmup  \
#     --source_path datamotion/datafactory --model_path datamotion/datafactory/output3000 --densify_until_iter 15000 --eval\
#     --resume  --pretrain_weights weights/bluryfactory/model_best.pth --save_dir ./logs/bluryfactory/3000

# ##blurypool
# # python3 train.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
# #     --train_ps 256 --train_dir datamotion/datapool\
# #     --val_ps 256 --val_dir datamotion/datapool --env _0706 \
# #     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro --warmup  \
# #     --source_path datamotion/datapool --model_path datamotion/datapool/output3000 --densify_until_iter 3000 --eval\
# #     --resume  --pretrain_weights weights/blurypool/model_best.pth --save_dir ./logs/blurypool/3000
# # # ##blurytanabata
# # python3 train.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
# #     --train_ps 256 --train_dir datamotion/datatanabata\
# #     --val_ps 256 --val_dir datamotion/datatanabata --env _0706 \
# #     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro --warmup  \
# #     --source_path datamotion/datatanabata --model_path datamotion/datatanabata/output3000 --densify_until_iter 15000 --eval\
# #     --resume  --pretrain_weights weights/blurytanabata/model_best.pth --save_dir ./logs/blurytanabata/3000
# # ##blurywine
# # python3 train.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
# #     --train_ps 256 --train_dir datamotion/datawine\
# #     --val_ps 256 --val_dir datamotion/datawine --env _0706 \
# #     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro --warmup  \
# #     --source_path datamotion/datawine --model_path datamotion/datawine/output3000 --densify_until_iter 15000 --eval  \
# #     --resume  --pretrain_weights weights/blurywine/model_best.pth --save_dir ./logs/blurywine/

# ######## DEFOCUS BLUR
# # # ##blurycozyroom
# python3 train.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datadefocus/datacozyroom \
#     --val_ps 256 --val_dir datadefocus/datacozyroom --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro   \
#     --source_path datadefocus/datacozyroom --model_path datadefocus/datacozyroom/output1000  --densify_until_iter 15000  --eval\
#     --resume  --pretrain_weights weights/defocus/blurcozyroom/model_best.pth --save_dir ./logs/blurcozyroom/3000 --train_iter 30000


# # # # # # ##blurfactory                                                
# python3 train.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datadefocus/datafactory\
#     --val_ps 256 --val_dir datadefocus/datafactory --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro   \
#     --source_path datadefocus/datafactory --model_path datadefocus/datafactory/output1000  --densify_until_iter 15000   --eval\
#     --resume  --pretrain_weights weights/defocus/blurfactory/model_best.pth --save_dir ./logs/blurfactory/3000
# # # # ### blur pool    
# python3 train.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datadefocus/datapool\
#     --val_ps 256 --val_dir datadefocus/datapool  --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro   \
#     --source_path datadefocus/datapool  --model_path datadefocus/datapool/output1000  --densify_until_iter 15000  --eval\
#     --resume  --pretrain_weights weights/defocus/blurpool/model_best.pth --save_dir ./logs/blurpool/3000
# python render.py -m datadefocus/datapool/output
# python metrics.py -m datadefocus/datapool/output

    

    
# # # # ### blur tanabata 
# python3 train.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datadefocus/datacozyroom\
#     --val_ps 256 --val_dir datadefocus/datacozyroom  --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro   \
#     --source_path datadefocus/datacozyroom  --model_path datadefocus/datacozyroom/outputu --eval --densify_until_iter 15000   \
#     --resume  --pretrain_weights weights/defocus/blurcozyroom/model_best.pth --save_dir ./logs/blurtanabata/3000 --train_iter 30000
# python render.py -m datadefocus/datacozyroom/outputu
# python metrics.py -m datadefocus/datacozyroom/outputu
# python3 train2s.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datadefocus/datacozyroom\
#     --val_ps 256 --val_dir datadefocus/datacozyroom  --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro   \
#     --source_path datadefocus/datacozyroom  --model_path datadefocus/datacozyroom/outputcs --eval --densify_until_iter 15000   \
#     --resume  --pretrain_weights weights/defocus/blurcozyroom/model_best.pth --save_dir ./logs/blurtanabata/3000 
# python render.py -m datadefocus/datacozyroom/outputcs
# python metrics.py -m datadefocus/datacozyroom/outputcs
# python3 train2s.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datadefocus1/datafactory\
#     --val_ps 256 --val_dir datadefocus1/datafactory  --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro   \
#     --source_path datadefocus1/datafactory  --model_path datadefocus1/datafactory/outputcs1   --densify_until_iter 3000   \
#     --resume  --pretrain_weights weights/defocus/blurfactory/model_best.pth --save_dir ./logs/blurtanabata/3000 
# python render.py -m datadefocus1/datafactory/outputcs1
# python metrics.py -m datadefocus1/datafactory/outputcs1
# python3 train2s.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datadefocus/datafactory\
#     --val_ps 256 --val_dir datadefocus/datafactory  --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro   \
#     --source_path datadefocus/datafactory  --model_path datadefocus/datafactory/outputcs1    --densify_until_iter 6000   \
#     --resume  --pretrain_weights weights/defocus/blurfactory/model_best3.pth --save_dir ./logs/blurtanabata/3000 
# python render.py -m datadefocus/datafactory/outputcs1
# python metrics.py -m datadefocus/datafactory/outputcs1
# python3 train.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datafactory\
#     --val_ps 256 --val_dir datafactory  --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro   \
#     --source_path datafactory  --model_path datafactory/output1000 --eval --densify_until_iter 3000   \
#     --resume  --pretrain_weights weights/defocus/blurfactory/model_best.pth --save_dir ./logs/blurtanabata/3000 
# python render.py -m datafactory/output1000
# python metrics.py -m datafactory/output1000
# python3 train.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datadefocus/datapool\
#     --val_ps 256 --val_dir datadefocus/datapool  --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro   \
#     --source_path datadefocus/datapool  --model_path datadefocus/datapool/output --eval --densify_until_iter 15000   \
#     --resume  --pretrain_weights weights/defocus/blurpool/model_best.pth --save_dir ./logs/blurtanabata/3000 --train_iter 30000
# python render.py -m datadefocus/datapool/output
# python metrics.py -m datadefocus/datapool/output

# python3 train2s.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datadefocus/datapool\
#     --val_ps 256 --val_dir datadefocus/datapool  --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro   \
#     --source_path datadefocus/datapool  --model_path datapool/outputcs --eval --densify_until_iter 3000   \
#     --resume  --pretrain_weights weights/defocus/blurpool/model_best.pth --save_dir ./logs/blurtanabata/3000 
# python render.py -m datapool/outputcs
# python metrics.py -m datapool/outputcs

# python3 train2s.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datadefocus/datatanabata\
#     --val_ps 256 --val_dir datadefocus/datatanabata  --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro   \
#     --source_path datadefocus/datatanabata  --model_path datadefocus/datatanabata/outputcs --eval --densify_until_iter 15000   \
#     --resume  --pretrain_weights weights/defocus/blurtanabata/model_best.pth --save_dir ./logs/blurtanabata/3000 
# python render.py -m datadefocus/datatanabata/outputcs
# python metrics.py -m datadefocus/datatanabata/outputcs

# python3 train2s.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datadefocus/datawine\
#     --val_ps 256 --val_dir datadefocus/datawine  --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro   \
#     --source_path datadefocus/datawine  --model_path datawine/outputcs --eval --densify_until_iter 15000   \
#     --resume  --pretrain_weights weights/defocus/blurwine/model_best.pth --save_dir ./logs/blurtanabata/3000 
# python render.py -m datawine/outputcs
# python metrics.py -m datawine/outputcs

# python3 train.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datadefocus/datawine\
#     --val_ps 256 --val_dir datadefocus/datawine  --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro   \
#     --source_path datadefocus/datawine  --model_path datadefocus/datawine/outputlam2  --densify_until_iter 15000 --lambda_dssim 0.2  --eval\
#     --resume  --pretrain_weights weights/defocus/blurwine/model_best.pth --save_dir ./logs/blurwine/3000
# # # # ### blur wine   
# python3 train.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datadefocus/datawine\
#     --val_ps 256 --val_dir datadefocus/datawine  --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro   \
#     --source_path datadefocus/datawine  --model_path datadefocus/datawine/output1000  --densify_until_iter 15000   --eval\
#     --resume  --pretrain_weights weights/defocus/blurwine/model_best.pth --save_dir ./logs/blurwine/3000
# python render.py -m datadefocus/datawine/output1000
# python metrics.py -m datadefocus/datawine/output1000


# python3 train.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datadefocus/datawine\
#     --val_ps 256 --val_dir datadefocus/datawine  --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro --warmup  \
#     --source_path datadefocus/datawine  --model_path datadefocus/datawine/output1000  --densify_until_iter 15000  --iteration 28000 --eval\
#     --resume  --pretrain_weights weights/defocus/blurwine/model_best.pth --save_dir ./logs/blurwine/3000

####REAL 
####MOTION
# python3 train2s.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datareal/motion/blurball\
#     --val_ps 256 --val_dir datareal/motion/blurball --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro --warmup  \
#     --source_path datareal/motion/blurball --model_path datareal/motion/blurball/output2s --eval --densify_until_iter 3000   \
#     --resume  --pretrain_weights weights/realmotion/Uformer_B.pth --save_dir ./logs/blurfactory/3000
# python render.py -m datareal/motion/blurball/output2s
# python metrics.py -m datareal/motion/blurball/output2s
# python trainorg.py -s datareal/motion/blurball -m datareal/motion/blurball/outputorg --eval
# python render.py -m datareal/motion/blurball/outputorg

# python3 train2s.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datareal/motion/blurball\
#     --val_ps 256 --val_dir datareal/motion/blurball --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro --warmup  \
#     --source_path datareal/motion/blurball --model_path datareal/motion/blurball/output2s15 --eval --densify_until_iter 15000   \
#     --resume  --pretrain_weights weights/realmotion/Uformer_B.pth --save_dir ./logs/blurfactory/3000
# python render.py -m datareal/motion/blurball/output2s15
# python metrics.py -m datareal/motion/blurball/output2s15

# python3 train2s.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datareal/motion/blurbasket\
#     --val_ps 256 --val_dir datareal/motion/blurbasket --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro --warmup  \
#     --source_path datareal/motion/blurbasket --model_path datareal/motion/blurbasket/output2s --eval --densify_until_iter 3000   \
#     --resume  --pretrain_weights weights/realmotion/Uformer_B.pth --save_dir ./logs/blurfactory/3000
# python render.py -m datareal/motion/blurbasket/output2s
# python metrics.py -m datareal/motion/blurbasket/output2s

# python3 train2s.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datareal/motion/blurbasket\
#     --val_ps 256 --val_dir datareal/motion/blurbasket --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro --warmup  \
#     --source_path datareal/motion/blurbasket --model_path datareal/motion/blurbasket/output2s15 --eval --densify_until_iter 15000   \
#     --resume  --pretrain_weights weights/realmotion/Uformer_B.pth --save_dir ./logs/blurfactory/3000
# python render.py -m datareal/motion/blurbasket/output2s15
# python metrics.py -m datareal/motion/blurbasket/output2s15
# python trainorg.py -s datareal/motion/blurbasket -m datareal/motion/blurbasket/outputorg --eval
# python render.py -m datareal/motion/blurbasket/outputorg

# python3 train2s.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datareal/motion/blurbuick\
#     --val_ps 256 --val_dir datareal/motion/blurbuick --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro --warmup  \
#     --source_path datareal/motion/blurbuick --model_path datareal/motion/blurbuick/output2so --eval --densify_until_iter 3000   \
#     --resume  --pretrain_weights weights/realmotion/Uformer_B.pth --save_dir ./logs/blurfactory/3000
# python render.py -m datareal/motion/blurbuick/output2so
# python metrics.py -m datareal/motion/blurbuick/output2so
# python trainorg.py -s datareal/motion/blurbuick -m datareal/motion/blurbuick/outputorg --eval
# python render.py -m datareal/motion/blurbuick/outputorg
# python trainorg0.py -s datareal/motion/blurbuick -m datareal/motion/blurbuick/outputorg0 --eval
# python render.py -m datareal/motion/blurbuick/outputorg0
# python3 train2s.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datareal/motion/blurbuick\
#     --val_ps 256 --val_dir datareal/motion/blurbuick --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro --warmup  \
#     --source_path datareal/motion/blurbuick --model_path datareal/motion/blurbuick/output2s6 --eval --densify_until_iter 6000   \
#     --resume  --pretrain_weights weights/realmotion/Uformer_B.pth --save_dir ./logs/blurfactory/3000
# python render.py -m datareal/motion/blurbuick/output2s6
# python metrics.py -m datareal/motion/blurbuick/output2s6
# python trainorg.py -s datareal/motion/blurgirl -m datareal/motion/blurgirl/outputorg --eval
# python render.py -m datareal/motion/blurgirl/outputorg

# python3 train2s.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datareal/motion/blurgirl\
#     --val_ps 256 --val_dir datareal/motion/blurgirl --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro --warmup  \
#     --source_path datareal/motion/blurgirl --model_path datareal/motion/blurgirl/output2s --eval --densify_until_iter 3000   \
#     --resume  --pretrain_weights weights/realmotion/Uformer_B.pth --save_dir ./logs/blurfactory/3000
# python render.py -m datareal/motion/blurgirl/output2s
# python metrics.py -m datareal/motion/blurgirl/output2s

# python trainorg0.py -s datareal/motion/blurgirl -m datareal/motion/blurgirl/outputorg0 --eval
# python render.py -m datareal/motion/blurgirl/outputorg0



# python3 train2s.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datareal/motion/blurgirl\
#     --val_ps 256 --val_dir datareal/motion/blurgirl --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro --warmup  \
#     --source_path datareal/motion/blurgirl --model_path datareal/motion/blurgirl/output2s1 --eval --densify_until_iter 1000   \
#     --resume  --pretrain_weights weights/realmotion/Uformer_B.pth --save_dir ./logs/blurfactory/3000
# python render.py -m datareal/motion/blurgirl/output2s1
# python metrics.py -m datareal/motion/blurgirl/output2s1



# python3 train2s.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datareal/motion/blurcoffee\
#     --val_ps 256 --val_dir datareal/motion/blurcoffee --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro --warmup  \
#     --source_path datareal/motion/blurcoffee --model_path datareal/motion/blurcoffee/output2s --eval --densify_until_iter 3000   \
#     --resume  --pretrain_weights weights/realmotion/Uformer_B.pth --save_dir ./logs/blurfactory/3000
# python render.py -m datareal/motion/blurcoffee/output2s
# python metrics.py -m datareal/motion/blurcoffee/output2s
# python trainorg.py -s datareal/motion/blurcoffee -m datareal/motion/blurcoffee/outputorg --eval
# python render.py -m datareal/motion/blurcoffee/outputorg

# python3 train2s.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datareal/motion/blurcoffee\
#     --val_ps 256 --val_dir datareal/motion/blurcoffee --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro --warmup  \
#     --source_path datareal/motion/blurcoffee --model_path datareal/motion/blurcoffee/output2s15 --eval --densify_until_iter 15000   \
#     --resume  --pretrain_weights weights/realmotion/Uformer_B.pth --save_dir ./logs/blurfactory/3000
# python render.py -m datareal/motion/blurcoffee/output2s15
# python metrics.py -m datareal/motion/blurcoffee/output2s15

# python3 train2s.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datareal/motion/blurdecoration\
#     --val_ps 256 --val_dir datareal/motion/blurdecoration --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro --warmup  \
#     --source_path datareal/motion/blurdecoration --model_path datareal/motion/blurdecoration/output2s --eval --densify_until_iter 3000   \
#     --resume  --pretrain_weights weights/realmotion/Uformer_B.pth --save_dir ./logs/blurfactory/3000
# python render.py -m datareal/motion/blurdecoration/output2s
# python metrics.py -m datareal/motion/blurdecoration/output2s
# python trainorg.py -s datareal/motion/blurdecoration -m datareal/motion/blurdecoration/outputorg --eval
# python render.py -m datareal/motion/blurdecoration/outputorg

# python3 train2s.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datareal/motion/blurheron\
#     --val_ps 256 --val_dir datareal/motion/blurheron --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro --warmup  \
#     --source_path datareal/motion/blurheron --model_path datareal/motion/blurheron/output2s --eval --densify_until_iter 3000   \
#     --resume  --pretrain_weights weights/realmotion/Uformer_B.pth --save_dir ./logs/blurfactory/3000
# python render.py -m datareal/motion/blurheron/output2s
# python metrics.py -m datareal/motion/blurheron/output2s
# python trainorg.py -s datareal/motion/blurheron -m datareal/motion/blurheron/outputorg --eval
# python render.py -m datareal/motion/blurheron/outputorg

# python3 train2s.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datareal/motion/blurparterre\
#     --val_ps 256 --val_dir datareal/motion/blurparterre --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro --warmup  \
#     --source_path datareal/motion/blurparterre --model_path datareal/motion/blurparterre/output2s --eval --densify_until_iter 3000   \
#     --resume  --pretrain_weights weights/realmotion/Uformer_B.pth --save_dir ./logs/blurfactory/3000
# python render.py -m datareal/motion/blurparterre/output2s
# python metrics.py -m datareal/motion/blurparterre/output2s
# python trainorg.py -s datareal/motion/blurparterre -m datareal/motion/blurparterre/outputorg --eval
# python render.py -m datareal/motion/blurparterre/outputorg

# python3 train2s.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datareal/motion/blurpuppet\
#     --val_ps 256 --val_dir datareal/motion/blurpuppet --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro --warmup  \
#     --source_path datareal/motion/blurpuppet --model_path datareal/motion/blurpuppet/output2s --eval --densify_until_iter 3000   \
#     --resume  --pretrain_weights weights/realmotion/Uformer_B.pth --save_dir ./logs/blurfactory/3000
# python render.py -m datareal/motion/blurpuppet/output2s
# python metrics.py -m datareal/motion/blurpuppet/output2s

# python3 train2s.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datareal/motion/blurpuppet\
#     --val_ps 256 --val_dir datareal/motion/blurpuppet --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro --warmup  \
#     --source_path datareal/motion/blurpuppet --model_path datareal/motion/blurpuppet/output2s15 --eval --densify_until_iter 15000   \
#     --resume  --pretrain_weights weights/realmotion/Uformer_B.pth --save_dir ./logs/blurfactory/3000
# python render.py -m datareal/motion/blurpuppet/output2s15
# python metrics.py -m datareal/motion/blurpuppet/output2s15
# python trainorg.py -s datareal/motion/blurpuppet -m datareal/motion/blurpuppet/outputorg --eval
# python render.py -m datareal/motion/blurpuppet/outputorg

# python3 train2s.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datareal/motion/blurstair\
#     --val_ps 256 --val_dir datareal/motion/blurheron --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro --warmup  \
#     --source_path datareal/motion/blurstair --model_path datareal/motion/blurstair/output2s --eval --densify_until_iter 3000   \
#     --resume  --pretrain_weights weights/realmotion/Uformer_B.pth --save_dir ./logs/blurfactory/3000
# python render.py -m datareal/motion/blurstair/output2s
# python metrics.py -m datareal/motion/blurstair/output2s
# python trainorg.py -s datareal/motion/blurstair -m datareal/motion/blurstair/outputorg --eval
# python render.py -m datareal/motion/blurstair/outputorg



#####DEFOCUS
# # # ##cake
# python3 train.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
#     --train_ps 256 --train_dir datareal/datadefocus/defocuscake\
#     --val_ps 256 --val_dir datareal/datadefocus/defocuscake --env _0706 \
#     --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro --warmup  \
#     --source_path datareal/datadefocus/defocuscake --model_path datareal/datadefocus/defocuscake/output0  --densify_until_iter 15000   --eval\
#     --resume  --pretrain_weights weights/defocus/custom/Uformer_B.pth --save_dir ./logs/blurfactory/3000
# python train2.py -s defocuscupcake -m defocuscupcake/output0 --eval 
# python render.py -m defocuscupcake/output0


