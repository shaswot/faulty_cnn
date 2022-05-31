import subprocess

experiment = "mnist32-cnn_1024_256_64-1023--LIM_500-2188"
model_instance, error_profile = experiment.split("--")

exp_script_list = [ 
                    "all-RowShuffle_oplayer_lenet_3hidden_ERRexpbitflips_2",
                    "all-RowShuffle_h2layer_lenet_3hidden_ERRexpbitflips_2",
                  ]

for script_name in exp_script_list:
    cmd_script = "exp_name="+script_name + " model_instance="+model_instance + " error_profile="+error_profile +" bash dispatch.sh"
    print(cmd_script)
    subprocess.call(cmd_script, shell=True)
    
