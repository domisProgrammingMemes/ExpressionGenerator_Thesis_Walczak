import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from CSVDataLoader import AUDataset
from torch.utils.tensorboard import SummaryWriter


# set up the divice (GPU or CPU) via input prompt if wanted
# Wird aktuell nicht benötigt
def set_device():
    cuda_true = input("Use GPU? (y) or (n)?")
    if cuda_true == "y":
        device = "cuda"
    else:
        device = "cpu"
    print("Device:", device)


# müssen aktuell 1 sein!
train_batch_size = 1
val_batch_size = 1
test_batch_size = 1

torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    # path of training dataset
    csv_read_path = r"Data\dataset"

    dataset = AUDataset(csv_read_path)
    trainset, valset, testset = torch.utils.data.random_split(dataset, [205, 25, 25])           # whole dataset = 255 samples


    train_loader = DataLoader(dataset=trainset, batch_size=train_batch_size, shuffle=True, num_workers=0,
                              drop_last=True)
    val_loader = DataLoader(dataset=valset, batch_size=val_batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(dataset=testset, batch_size=test_batch_size, shuffle=True, num_workers=0, drop_last=True)


    class ExpressionGenerator(nn.Module):
        def __init__(self, n_features: int, n_output_encoder: int, n_hidden: int, n_layers: int, p: float):
            super(ExpressionGenerator, self).__init__()
            self.n_features = n_features
            self.hidden_size = n_hidden
            self.n_layers = n_layers
            self.dropout = p
            self.output_encoder_size = n_output_encoder

            # hidden and cell for LSTM
            self.hidden = torch.zeros(self.n_layers, 1, self.hidden_size).to(device)
            self.cell = torch.zeros(self.n_layers, 1, self.hidden_size).to(device)

            # encoding lstm
            self.encoding = nn.LSTM(input_size=n_features, hidden_size=n_output_encoder, num_layers=n_layers, dropout=p)
            self.linear = nn.Linear(n_output_encoder, n_features)

            # lstm - temporal information
            self.rnn = nn.LSTM(n_output_encoder + n_features, n_hidden, num_layers=n_layers, batch_first=True)

            # decoder ->
            self.decoding = nn.Linear(n_hidden, n_features)


        def forward(self, x, target, hidden, cell):
            """
            :param x: ist target besteht aus 20frames
            :param target: ist das Ende der sequenz
            :return: prediction in Form eines einzelnen Frames
            """
            # variables (dimensions): 15, 256, 512, 1

            # encoding
            _, (hidden, cell) = self.encoding(x, (hidden, cell))
            # encoding = self.linear(hidden)                      # encoding size [1, 1, 15]
            feature = torch.cat([hidden, target], dim=2)

            # temporal dynamics
            frame_encoding, (self.hidden, self.cell) = self.rnn(feature, (self.hidden, self.cell))

            # decoding with fc
            prediction = self.decoding(frame_encoding)

            return prediction, hidden, cell

        def zero_hidden(self):
            self.hidden = torch.zeros(self.n_layers, 1, self.hidden_size).to(device)
            self.cell = torch.zeros(self.n_layers, 1, self.hidden_size).to(device)

        def zero_hidden_encoding(self):
            hidden = torch.zeros(self.n_layers, 1, self.output_encoder_size).to(device)
            cell = torch.zeros(self.n_layers, 1, self.output_encoder_size).to(device)
            return hidden, cell


    # Hyperparameters
    num_epochs = 50
    learning_rate = 1e-3
    dropout = 0.5                               # not used right now
    teacher_forcing_ratio = 0.5                 # not used right now

    # safe path for models (best and last)
    best = "./models/ExGen_Best.pth"
    last = "./models/ExGen_Last.pth"

    # model
    model = ExpressionGenerator(15, 256, 512, 1, dropout)
    model.load_state_dict(torch.load(best))
    model = model.to(device)

    # define loss(es) and optimizer
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-6, amsgrad=True)

    ################################################################################################

    def fifo(tensor, x):
        """
        :param tensor: tensor welche als Grundlage des FIFO Prinzips gilt
        :param x: Wert welcher an letzter Stelle angehängt werden soll
        :return: verschobener Tensor mit x an letzter Stelle
        """
        return torch.cat((tensor[1:], x))

    # TRAINING

    # best current test error (MSE):
    best_error = 10000  # for first iteration arbitrary high
    last_epoch = 0


    def train_model(train: DataLoader, val: DataLoader, n_Epochs: int, best_test_error: float):
        best_epoch = 0
        writer = SummaryWriter()
        loss_history = []
        print("Start training...")
        for epoch in range(1 + last_epoch, n_Epochs + 1 + last_epoch):
            model.train()
            train_loss = 0
            for index, data in enumerate(train):
                batch_data, name = data
                batch_data = batch_data.to(device)
                seq_length = batch_data.size(1)
                number_aus = batch_data.size(2)

                # create empty sequence tensor for whole anim:
                created_sequence = torch.zeros(1, seq_length, number_aus).to(device)

                sequence = batch_data[0, 0:20]
                # first 20 frames are known and should be copied
                for i in range(0, 20):
                    created_sequence[0][i] = sequence[i]

                sequence = sequence.unsqueeze(1)
                last_frame = batch_data[0, -1]
                last_frame = last_frame.unsqueeze(0)
                last_frame = last_frame.unsqueeze(0)

                # für jede Sequenz
                optimizer.zero_grad()
                model.zero_hidden()
                hidden, cell = model.zero_hidden_encoding()

                for t in range(20, seq_length):
                    prediction, hidden, cell = model(sequence, last_frame, hidden, cell)

                    created_sequence[0][t] = prediction

                    # teacher forcing falls damit trainiert werden möchte
                    # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
                    # if use_teacher_forcing:
                    #     # if teacher forced give it the correct next frame as part of target:
                    #     target = torch.cat([real_next_frame, last_frame]).to(device)
                    #     # print("with teacher force", target.size())
                    #     # exit()
                    # else:
                    #     # neues ziel da neuer frame:
                    #     target = torch.cat([prediction, last_frame]).to(device)
                    #     # print("without teacher force", target.size())
                    #     # exit()

                    # define new target
                    sequence = fifo(sequence, prediction)

                # loss calculation
                loss = mse_loss(created_sequence, batch_data)
                loss.backward()

                # grad clipping -> caused cuda-error
                # nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                # upgrade gradients
                optimizer.step()
                train_loss = train_loss + loss.item()

            # eval the model on the val set
            model.eval()
            with torch.no_grad():
                val_loss_mse = 0
                for index, data in enumerate(val):
                    batch_data, name = data
                    batch_data = batch_data.to(device)
                    seq_length = batch_data.size(1)
                    number_aus = batch_data.size(2)

                    # create empty sequence tensor for whole anim:
                    created_sequence = torch.zeros(1, seq_length, number_aus).to(device)

                    sequence = batch_data[0, 0:20]
                    # first 20 frames are known and should be copied
                    for i in range(0, 20):
                        created_sequence[0][i] = sequence[i]

                    sequence = sequence.unsqueeze(1)
                    last_frame = batch_data[0, -1]
                    last_frame = last_frame.unsqueeze(0)
                    last_frame = last_frame.unsqueeze(0)

                    # für jede sequenz
                    optimizer.zero_grad()
                    model.zero_hidden()
                    hidden, cell = model.zero_hidden_encoding()

                    for t in range(20, seq_length):
                        prediction, hidden, cell = model(sequence, last_frame, hidden, cell)
                        created_sequence[0][t] = prediction
                        # define new target
                        sequence = fifo(sequence, prediction)

                    loss_mse = mse_loss(created_sequence, batch_data)
                    val_loss_mse = val_loss_mse + loss_mse.item()

            print(f"Epoch {epoch} of {num_epochs + last_epoch} epochs - Train (MSE): {train_loss:.4f} --- Val (MSE) = {val_loss_mse:.4f}")

            # save data to tensorboard and txt!
            loss_history.append(train_loss)
            writer.add_scalar("MSE_Loss - train", train_loss, epoch)
            writer.add_scalar("MSE_Loss - val", val_loss_mse, epoch)

            if val_loss_mse < best_test_error:
                torch.save(model.state_dict(), best)
                best_test_error = val_loss_mse
                best_epoch = epoch
                print("New Model had been saved!")


        # append to txt .. better save than sorry!
        with open(r'training_history\NewFile', 'a') as f:
            print(loss_history, file=f)

        writer.close()
        print("Best test error (for copy-paste):", best_test_error)
        print("Epoch (best test error):", best_epoch)
        print("Finished training!")
        torch.save(model.state_dict(), last)


    def test_model(test: DataLoader):
        model.eval()
        with torch.no_grad():
            test_loss_mse = 0
            for index, data in enumerate(test):
                batch_data, name = data
                batch_data = batch_data.to(device)
                # print(batch_data.size())      # size = [batch_size, sequence_length, n_features]
                # print(name)                   # csv name
                seq_length = batch_data.size(1)
                number_aus = batch_data.size(2)

                # create empty sequence tensor for whole anim:
                created_sequence = torch.zeros(1, seq_length, number_aus).to(device)

                sequence = batch_data[0, 0:20]
                # first 10 frames are known and should be copied
                for i in range(0, 20):
                    created_sequence[0][i] = sequence[i]

                sequence = sequence.unsqueeze(1)  # add dimension for batch [10, 1, 15]
                # first_frame = batch_data[0, 0]
                last_frame = batch_data[0, -1]
                last_frame = last_frame.unsqueeze(0)
                last_frame = last_frame.unsqueeze(0)

                # für jede sequenz
                optimizer.zero_grad()
                model.zero_hidden()
                hidden, cell = model.zero_hidden_encoding()

                for t in range(20, seq_length):
                    prediction, hidden, cell = model(sequence, last_frame, hidden, cell)
                    created_sequence[0][t] = prediction
                    # define new target
                    sequence = fifo(sequence, prediction)

                loss_mse = mse_loss(created_sequence, batch_data)
                test_loss_mse = test_loss_mse + loss_mse.item()


        print(f"Test_losses: MSE = {test_loss_mse:.4f}")
        with open(r'test_history\NewFile', 'a') as f:
            print(f"MSE:{test_loss_mse}", file=f)

    # train_model(train_loader, val_loader, num_epochs, best_error)
    # test_model(test_loader)

    ################################################################################################

    # GENERATION
    # which model to load from path
    best = "./models/ExGen_Best.pth"
    # where to save the animation/wo sollen die Animationen gespeichert werden?
    gen_save_pth = "./Generation/"

    """
    ! Hier könnte man das Netz alternativ erstellen und das beste Modell laden !
    """
    # model = ExpressionGenerator(15, 256, 512, 1, dropout)
    # model.load_state_dict(torch.load(best))
    # model = model.to(device)


    def sequenize_start(start_frame: torch.Tensor):
        """
        Funktion welche den ersten Frame 20 kopiert und in einen Tensor verpackt
        :param start_frame: erster Frame der Animation
        :return: sequenz mit 20 mal dem ersten Frame
        Dies wird in der Funktion "generate_expression" aufgerufen
        """
        sequence = torch.empty(20, 15)
        for i in range(0, 20):
            sequence[i] = start_frame

        return sequence

    def generate_expression(start_frame: torch.Tensor, end_frame: torch.Tensor, sequence_length: int, anim_name: str):
        """
        Funktion mit welcher Animationen generiert werden können
        :param start_frame: der erste Frame der Animation
        :param end_frame: der letzte Frame der Animation
        :param sequence_length: länge der Sequenz (diese sollte länger als bei den Trainingsdaten sein,
                                da das Netz sonst eventuell nicht die ganze Animation generiert)
        :param anim_name: Name unter welchem die Animation später gefunden werden kann
        :return: no return
        """
        # eval the model on the test set
        model.eval()
        with torch.no_grad():


            last_frame = end_frame.to(device)
            last_frame = last_frame.unsqueeze(0)
            last_frame = last_frame.unsqueeze(0)
            number_aus = end_frame.size(0)

            created_sequence = torch.zeros(1, sequence_length, number_aus).to(device)

            sequence = sequenize_start(start_frame).to(device)
            # first 20 frames are known and should be copied
            for i in range(0, 20):
                created_sequence[0][i] = sequence[i]

            sequence = sequence.unsqueeze(1)  # add dimension for batch [20, 1, 15]

            model.zero_hidden()
            hidden, cell = model.zero_hidden_encoding()

            for t in range(20, sequence_length):
                prediction, hidden, cell = model(sequence, last_frame, hidden, cell)
                created_sequence[0][t] = prediction
                # define new target
                sequence = fifo(sequence, prediction)

            # for convenience:
            sequence = created_sequence

            sequence = sequence.cpu()
            sequence = sequence.squeeze(0)

            # get right format for columns
            df = pd.read_csv(csv_read_path + "/neutralhappy2_fill.csv")
            header = list(df.drop(["Frame"], axis=1))
            del df

            # generate new name for the generated animation
            new_name = "ExGen_" + str(anim_name)

            # transform predictions to csv
            sequence_np = sequence.numpy()
            sequence_df = pd.DataFrame(sequence_np)
            sequence_df.columns = header
            # prediction_df.columns = ["AU1L","AU1R","AU2L","AU2R","AU4L","AU4R","AU6L","AU6R","AU9","AU10","AU13L","AU13R","AU18","AU22","AU27"]
            sequence_df.to_csv(gen_save_pth + new_name + ".csv")
            del sequence_np
            del sequence_df


    # DATA
    # happy_frown5 -> frown
    frown = torch.Tensor([0.11600816761214412, 0.1284655902491436, 0.04555017236853346, 0.07891883398256783, 1.1624704923767863,1.1292259224988748, 4.1179628134090425e-09, 1.3139848525048626e-08, 0.3780972373525189,0.002627721186276977, 0.4052149252538391, 0.006900503041645656, 2.656951757012366e-10,-8.446394673406755e-10, -9.975166643120387e-10])
    # surprise_disgust2 -> surprise
    surprise = torch.Tensor([1.1999999112205617,1.1999999159976815,1.1999999328538695,1.1488722636177362,4.595641601005289e-08,1.7807415136001172e-07,4.928640608924983e-05,3.610486793096247e-05,9.383047183030742e-08,-1.3288099099324539e-08,2.009467533535717e-08,-1.4430244360720015e-09,-2.4141230449188912e-08,5.907091004922681e-09,1.2000000099788897])
    # happy_sad5 -> sad
    sad = torch.Tensor([0.1469748339078054, 0.1645827189561475, 0.09715293786052676, 0.09448502011811992,1.0443333135088018,1.1510849319027174, 0.045217939313237115, 0.20751686582277729,0.0356037971287012, 2.5040262983977084e-08,0.024025351834480672, 0.00011304006851385568,0.17286549919783545, 4.5758218435735396e-08,1.1034485458071822e-08])
    # happy_neutral -> neutral
    neutral = torch.Tensor([1.3183410608020914e-09, 0.007341170604253437, 6.560640186486611e-10, 6.156423513747244e-10, 0.05534238495332509, 0.03196604997678923, 4.029073564969686e-07, 0.05242904922403188, 2.121624087550293e-09, 3.935684714177853e-09, 4.39079980658653e-12, 0.06648207797258246, 0.08107495649839419, 4.873076503851266e-10,0.02053503309248228])
    # happy_neutral -> happy
    happy = torch.Tensor([1.7312022381517743e-07,9.942494218012084e-07,5.5279378574193394e-08,7.275862012056518e-08,0.1614868551455088,0.10531355676472627,1.199833379749608,1.1999999171354383,6.253415944733785e-08,0.25089657727780995,0.5664661956248268,1.0567180085991217,5.8532759285127815e-09,4.522632508438027e-09,1.2681872116221288e-07])
    # surprise_disgust -> disgust
    disgust = torch.Tensor([1.0670191593715594e-07,3.2528796729084693e-07,0.12982850058516537,0.06965097196072645,1.199999695066714,1.1706918609055257,0.2146797932278581,0.046555729206558386,0.042598584997725494,0.4542929713122771,0.401825638510415,8.596737489378636e-07,4.7000663724022786e-08,1.2655896810496993e-08,1.0064052277759405e-08])


    mydict = {
        "disgust": disgust,
        "frown": frown,
        "happy": happy,
        "neutral": neutral,
        "surprise": surprise,
        "sad": sad
    }

    def generate_multiple():
        from itertools import permutations
        perm = permutations(mydict, 2)
        for i in list(perm):
            x, y = i
            name = "i_" + x + "2" + y
            generate_expression(mydict[x], mydict[y], 500, name)

    # generate_multiple()




