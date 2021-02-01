import mirdata
import numpy as np
import sklearn
import random
import torch
import torchaudio
import pytorch_lightning as pl


class MridangamDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mirdataset,
        seq_duration=0.5,
        random_start=True,
        resample=8000,
        subset=0,
        train_split=0.8,
        test_split=0.2,
        random_seed=42
    ):
        """
        """
        self.seq_duration = seq_duration
        self.dataset = mirdataset
        self.track_ids = self.dataset.track_ids
        self.tracks = self.dataset.load_tracks()
        self.resample = resample
        self.set = subset
        self.random_start = random_start

        labels = [self.dataset.track(i).stroke_name for i in self.track_ids]
        unique_labels = list(set(labels)) ### unique labels
        self.labels = {label:i for i,label in enumerate(unique_labels)}
        self.trackids_train, self.trackids_test = sklearn.model_selection.train_test_split(self.track_ids, train_size=1-test_split, random_state=random_seed, stratify=labels)
        self.trackids_train, self.trackids_valid = sklearn.model_selection.train_test_split(self.track_ids, train_size=train_split, random_state=random_seed, stratify=labels)


    def __getitem__(self, index):

        if self.set==0:
            track_id = self.trackids_train[index]
        elif self.set==1:
            track_id = self.trackids_valid[index]
        elif self.set==2:
            track_id = self.trackids_test[index]
        track = self.dataset.track(track_id)

        #### compute start and end frames to read from the disk
        si, ei = torchaudio.info(track.audio_path)
        sample_rate, channels, length = si.rate, si.channels, si.length
        duration = length / sample_rate
        if self.seq_duration>duration:
            offset = 0
            num_frames = length
        else:
            if self.random_start:
                start = random.uniform(0, duration - self.seq_duration)
            else:
                start = 0.
            offset = int(np.floor(start * sample_rate))
            num_frames = int(np.floor(self.seq_duration * sample_rate))


        #### get audio
        audio_signal, sample_rate = torchaudio.load(filepath=track.audio_path, offset=offset,num_frames=num_frames)

        #### zero pad if the size is smaller than seq_duration
        seq_duration_samples = int(self.seq_duration * sample_rate)
        total_samples = audio_signal.shape[-1]
        if seq_duration_samples>total_samples:
            audio_signal = torch.nn.ConstantPad2d((0,seq_duration_samples-total_samples,0,0),0)(audio_signal)

        #### resample
        audio_signal = torchaudio.transforms.Resample(sample_rate, self.resample)(audio_signal)

        return audio_signal, self.labels[track.stroke_name]

    def __len__(self):
        if self.set==0:
            return len(self.trackids_train)
        elif self.set==1:
            return len(self.trackids_valid)
        else:
            return len(self.trackids_test)

class M5(pl.LightningModule):
    '''
    M5 neural net taken from: https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html
    '''
    def __init__(self, n_input=1, n_output=10, stride=8, n_channel=32):
        super().__init__()
        #### network
        self.conv1 = torch.nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = torch.nn.BatchNorm1d(n_channel)
        self.pool1 = torch.nn.MaxPool1d(4)
        self.conv2 = torch.nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = torch.nn.BatchNorm1d(n_channel)
        self.pool2 = torch.nn.MaxPool1d(4)
        self.conv3 = torch.nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = torch.nn.BatchNorm1d(2 * n_channel)
        self.pool3 = torch.nn.MaxPool1d(4)
        self.conv4 = torch.nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = torch.nn.BatchNorm1d(2 * n_channel)
        self.pool4 = torch.nn.MaxPool1d(4)
        self.fc1 = torch.nn.Linear(2 * n_channel, n_output)

        #### metrics
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        self.test_cm = pl.metrics.classification.ConfusionMatrix(num_classes=n_output)


    def forward_function(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = torch.nn.functional.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = torch.nn.functional.relu(self.bn4(x))
        x = self.pool4(x)
        # x = torch.nn.functional.avg_pool1d(x) #, kernel_size=x.shape[-1],stride=1
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return torch.nn.functional.log_softmax(x, dim=2).squeeze(1)

    def training_step(self, batch, batch_idx):
        waveform, label = batch
        output = self.forward_function(waveform)
        ### why log softmax and nll loss: https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/
        loss = torch.nn.functional.nll_loss(output, label)
        self.log('train_loss', loss)
        self.train_acc(output, label)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        waveform, label = batch
        output = self.forward_function(waveform)
        loss = torch.nn.functional.nll_loss(output, label)
        self.log('val_loss', loss)
        self.valid_acc(output, label)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        waveform, label = batch
        output = self.forward_function(waveform)
        loss = torch.nn.functional.nll_loss(output, label)
        self.log('test_loss', loss)
        self.test_acc(output, label)
        self.log('test_acc', self.test_acc, on_step=True, on_epoch=False)
        self.test_cm(output, label)

    def training_epoch_end(self, outputs):
        # log epoch metric
        self.log('train_acc', self.train_acc.compute(), prog_bar=True)

    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.valid_acc.compute(), prog_bar=True)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=4e-2,weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # reduce the learning after 10 epochs by a factor of 10
        return [optimizer], [scheduler]


#### Init the Mridangam stroke dataset
data_home='/Volumes/Macintosh HD 2/Documents/git/mirdata/tests/resources/mir_datasets_full/mridangam_stroke'
mridangam = mirdata.initialize("mridangam_stroke") #,data_home=data_home

download = False
if download:
    mridangam.download()

random_seed=0
pl.utilities.seed.seed_everything(seed=random_seed)

#### Pytorch dataset loaders
train_dataset = MridangamDataset(mirdataset=mridangam,subset=0, random_seed=random_seed)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=128,num_workers=24,pin_memory=True)
valid_dataset = MridangamDataset(mirdataset=mridangam,subset=1, random_seed=random_seed,pin_memory=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=128,num_workers=24,pin_memory=True)
test_dataset = MridangamDataset(mirdataset=mridangam,subset=2, random_seed=random_seed)
test_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=128,num_workers=24,pin_memory=True)

### Which batch size/learning rate?
# Theory suggests that when multiplying the batch size by k, one should multiply the learning rate by sqrt(k) to keep the variance in the gradient expectation constant. See page 5 at A. Krizhevsky. One weird trick for parallelizing convolutional neural networks: https://arxiv.org/abs/1404.5997

# However, recent experiments with large mini-batches suggest for a simpler linear scaling rule, i.e multiply your learning rate by k when using mini-batch size of kN. See P.Goyal et al.: Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour https://arxiv.org/abs/1706.02677

#### Initialize the model
model = M5(n_input=train_dataset[0][0].shape[0], n_output=len(train_dataset.labels))

#### Initialize a trainer
trainer = pl.Trainer(gpus=1, max_epochs=3, progress_bar_refresh_rate=5)

#### Train the model
trainer.fit(model, train_loader, valid_loader)

#### Compute metrics on the test set
trainer.test(test_dataloaders=test_loader)

#### Compute confusion matrix on the test set
confusion_matrix = model.test_cm.compute().cpu().numpy()
import matplotlib.pyplot as plt
plt.matshow(confusion_matrix)
plt.show()

import pdb;pdb.set_trace()




