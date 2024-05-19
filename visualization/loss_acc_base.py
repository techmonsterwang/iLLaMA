import matplotlib.pyplot as plt


# 读取log103.txt文件中的数据
with open('/mnt/petrelfs/wangjiahao/DoiT/output/103_doit7_base_basic/log.txt', 'r') as file:
    lines = file.readlines()

test_acc1_103 = []
train_loss_103 = []
epoch_103 = []

for line in lines:
    data = eval(line)
    test_acc1_103.append(data['test_acc1'])
    train_loss_103.append(data['train_loss'])
    epoch_103.append(data['epoch'])

# 读取log105.txt文件中的数据
with open('/mnt/petrelfs/wangjiahao/DoiT/output/148_doit16_base_basic/log.txt', 'r') as file:
    lines = file.readlines()

test_acc1_105 = []
train_loss_105 = []
epoch_105 = []

for line in lines:
    data = eval(line)
    test_acc1_105.append(data['test_acc1'])
    train_loss_105.append(data['train_loss'])
    epoch_105.append(data['epoch'])

# 绘制折线图
fig, ax1 = plt.subplots(figsize=(16, 8))
# plt.rcParams['font.family'] = 'New Times Roman'
plt.rcParams.update({'font.size': 18})  # 设置字体大小

ax1.set_xlabel('epochs', color='black', fontsize=24)
ax1.set_ylabel('accuracy (%)', color='black', fontsize=24)
ax1.plot(epoch_103, test_acc1_103, linestyle='-', color='#0071BC', label='baseline test acc')
ax1.plot(epoch_105, test_acc1_105, linestyle='-', color='#EB9B78', label='soft mask test acc')
ax1.tick_params(axis='x', labelcolor='black', labelsize=24)
ax1.tick_params(axis='y', labelcolor='black', labelsize=24)

ax2 = ax1.twinx()
ax2.set_ylabel('train loss', color='black', fontsize=24)
ax2.plot(epoch_103, train_loss_103, linestyle='--', color='#0071BC', label='baseline train loss')
ax2.plot(epoch_105, train_loss_105, linestyle='--', color='#EB9B78', label='soft mask train loss')
ax2.tick_params(axis='y', labelcolor='black', labelsize=24)

ax1.set_ylim(30, 90.0)
ax1.set_yticks([40,50,60,70,80,90])
ax2.set_ylim(2.5, 6.5)
ax2.set_yticks([3,4,5,6])
ax1.set_xlim(0, 300)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='right')

plt.savefig('train_loss_base.png', dpi=300)
plt.savefig('train_loss_base.pdf', dpi=300)
plt.savefig('train_loss_base.svg', dpi=300)
plt.show()