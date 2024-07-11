from metrics import compute_bler as bler
from metrics import BitErrorRate as ber


def train_dis(model, train_loader, optimizer, epoch, LR):
    t = time.time()
    
    for batch_idx, (m, x, z, y, magnitude, syndrome) in enumerate(train_loader):
        with tf.GradientTape() as tape:
            loss = model.train(x)
            
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if (batch_idx + 1) % 10 == 0 or batch_idx == len(train_loader) - 1:
            print(f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.5e}')
            
    print(f'Epoch {epoch} Train Time {time.time() - t}s\n')
            

def test_dis(model, test_loader_list, EbNo_range_test, min_FER=100, max_cum_count=1e7, min_cum_count=1e5):
    printed = False
    ber_list, bler_list = [], []

    for ix, test_loader in enumerate(test_loader_list):
        for batch_ix, (m, c, z, r, _, _, magnitude, syndrome) in enumerate(test_loader):
            c_hat = model(r)
            ber_list.append(ber(c, c_hat))
            bler_list.append(bler(c, c_hat))
            print(f'Test EbN0={EbNo_range_test[ix]}, BER={ber_list[-1]}')
            print(f'Test EbN0={EbNo_range_test[ix]}, BLER={bler_list[-1]}')

            if not printed:
              print("c: ", c)
              print("c_hat: ", c_hat)
              printed = True
            break

    return { "ber": ber_list, "bler": bler_list }


def test_models(): 
    data = { "LTDM": dict() } # only decoders

    for ix, tst_dataset in enumerate(test_ebnos_datasets):
        print(f"\nTesting on {list(dataset_types.keys())[ix].upper()}")
        
        print("Testing  Linear Transformer Diffusion Model...")
        data["LTDM"][ix] = test(dec, tst_dataset, EbNo_range_test, min_FER=50,max_cum_count=1e6,min_cum_count=1e4)
        return data



#                         z_cw   m 1s   1-cw     Should use zero codeword by default
dataset_types = {
              "rnd_bits":(False, False, False), # Binary bits sent and recieved with some awgn
              "flip_cw": (True, False, True),   # Zero codeword flipped to a all ones vector [1,1,...,1]
              "zero_cw": (True, False, False),  # Standard zero codeword used for training
              "ones_m":  (False, True, False),  # Makes the message all ones vector and passes it to generator matrix producing codeword and pcm
              }
