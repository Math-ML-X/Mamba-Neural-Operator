import torch

# check torch version
print(torch.__version__)

flag = torch.cuda.is_available()
print(flag)

ngpu = 4
# Decide which device we want to run on
for i in range(torch.cuda.device_count()):
    print("GPU {}: {}".format(i, torch.cuda.get_device_name(i)))

    device = torch.device(f"cuda:{i}" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    print(device)
    print(torch.cuda.get_device_name(0))
    print(torch.rand(3, 3).to(device))


