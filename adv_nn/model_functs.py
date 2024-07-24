from sionna.utils.metrics import compute_ber, compute_bler
import tensorflow as tf
import time


def train_gen():
    pass


def test_gen():
    pass


# dec5G = LDPC5GDecoder(enc5G, args, return_llrs5g=True)
# dec = Decoder(args)
def train_dec_5G(dec5G, epoch, training_len=100):
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    t = time.time()

    for batch_idx in range(training_len):
        u = binary_source([1, args.k])
        c = enc5G(u) # (1,n)

        x = mapper(c) # map c to symbols x
        y = channel([x, no]) # transmit over AWGN channel
        llr_ch = demapper([y, no]) # demap y to LLRs (1,n)

        if dec5G.return_llrs5g:
            llr_5g = dec5G(llr_ch)

            loss = dec.train_step(llr_5g)

        else:
            c_hat = dec5G(llr_ch)
            print("c, c_hat: ", c, c_hat)
            loss = loss_fn(c_hat, c)

        if (batch_idx + 1) % 1 == 0:
            print(f'Training epoch {epoch}, Batch {batch_idx + 1}/{training_len}, Loss={loss.numpy():.5e}')

    print(f'Epoch {epoch} Train Time {time.time() - t}s\n')


# dec5G = LDPC5GDecoder(enc5G, args, return_llrs5g=False, return_infobits=True)
def test_dec_5G(dec5G, no, testing_len=100):
    # printed = False
    ber_list, bler_list = [], []

    for batch_idx in range(testing_len):
        u = binary_source([1, args.k])
        c = enc5G(u) # (1,n)

        x = mapper(c) # map c to symbols x
        y = channel([x, no]) # transmit over AWGN channel
        llr_ch = demapper([y, no]) # demap y to LLRs (1,n)

        if dec5G._return_infobits: 
            u_hat = dec5G(llr_ch)
            ber_list.append( compute_ber(u, u_hat) ) # BER
            bler_list.append( compute_bler(u, u_hat) ) # BLER
        else: # return cw
            c_hat = dec5G(llr_ch)
            ber_list.append( compute_ber(c, c_hat) ) # BER
            bler_list.append( compute_bler(c, c_hat) ) # BLER

        print(f'Test EbN0={no}, BER={ber_list[-1]}')
        print(f'Test EbN0={no}, BLER={bler_list[-1]}')

        # if not printed:
        #     tensor, pred = u, u_hat if dec5G._return_infobits else c, c_hat
        #     print(f"info_tensor: {tensor}, pred: {pred}")
        #     printed = True
        # break

    return { "ber": ber_list, "bler": bler_list }


def train_dec(model, train_loader, optimizer, epoch, LR, traindata_len):
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    t = time.time()

    print(train_loader)
    for batch_idx, (_, x, _, _, _, _, _, _) in enumerate(train_loader):
        with tf.GradientTape() as tape:
            z_hat, z_mul, c_t = model.train(x)
            loss = loss_fn(z_hat, z_mul)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if (batch_idx + 1) % 10 == 0 or batch_idx == traindata_len - 1:
            print(f'Training epoch {epoch}, Batch {batch_idx + 1}/{traindata_len}: LR={LR:.2e}, Loss={loss.numpy():.5e}')

    print(f'Epoch {epoch} Train Time {time.time() - t}s\n')
            

def test_dec(model, test_loader_list, EbNo_range_test, min_FER=100, max_cum_count=1e7, min_cum_count=1e5):
    printed = False
    ber_list, bler_list = [], []

    for ix, test_loader in enumerate(test_loader_list):
        for batch_ix, (m, c, z, r, _, _, magnitude, syndrome) in enumerate(test_loader):
            c_hat, z_hat, dif_iter = model(r)
            print(c_hat.shape)
            ber_list.append( compute_ber(c, c_hat) ) # BER
            bler_list.append( compute_bler(c, c_hat) ) # BLER
            print(f'Test EbN0={EbNo_range_test[ix]}, BER={ber_list[-1]}')
            print(f'Test EbN0={EbNo_range_test[ix]}, BLER={bler_list[-1]}')

            if not printed:
                print("c: ", c)
                print("c_hat: ", c_hat)
                printed = True
            break

    return { "ber": ber_list, "bler": bler_list }


def test_models(model, test_ebnos_datasets, EbNo_range_test):
    data = { "LTDM": dict() } # only decoders

    for ix, tst_dataset in enumerate(test_ebnos_datasets):
        print(f"\nTesting on {list(dataset_types.keys())[ix].upper()}")

        print("Testing  Linear Transformer Diffusion Model...")
        data["LTDM"][ix] = test_dec(model, tst_dataset, EbNo_range_test, min_FER=50,max_cum_count=1e6,min_cum_count=1e4)
        return data



#                         z_cw   m 1s   1-cw     Should use zero codeword by default
dataset_types = {
              "rnd_bits":(False, False, False), # Binary bits sent and recieved with some awgn
              "flip_cw": (True, False, True),   # Zero codeword flipped to a all ones vector [1,1,...,1]
              "zero_cw": (True, False, False),  # Standard zero codeword used for training
              "ones_m":  (False, True, False),  # Makes the message all ones vector and passes it to generator matrix producing codeword and pcm
              }
