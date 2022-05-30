import subprocess

experiment = "mnist32-cnn_1024_256_64-1023--LIM_01-2188"
model_instance, error_profile = experiment.split("--")

exp_script_list = [ 
                    "all-RowShuffle_c0layer_lenet_3hidden_ERRexpbitflips_-1",
                    "all-RowShuffle_h2layer_lenet_3hidden_ERRexpbitflips_-1",
                  ]

for script_name in exp_script_list:
    cmd_script = "exp_name="+script_name + " model_instance="+model_instance + " error_profile="+error_profile +" bash dispatch.sh"
    print(cmd_script)
    subprocess.call(cmd_script, shell=True)
    
