import numpy as np
import tensorflow as tf
import time
import os

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


class Trainer:

    def __init__(self, model, model_dir):
        self.model = model
        self.model_dir = model_dir
        self.out_dir = model.config['out_dir']
        self.train_all = []
        self.test_all = []

    def _save_checkpoint(self, saver, sess, checkpoint_path, log, retries=8):
        meta_path = checkpoint_path + '.meta'
        write_meta_graph = not os.path.exists(meta_path)
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        for attempt in range(retries):
            try:
                saver.save(sess, checkpoint_path, write_meta_graph=write_meta_graph)
                return
            except tf.errors.OpError as exc:
                is_last_attempt = attempt == retries - 1
                if is_last_attempt:
                    raise

                log('  -> Checkpoint save retry {}/{} after TensorFlow file error: {}'.format(
                    attempt + 1, retries - 1, exc
                ))

                # On Windows, antivirus/sync tools can momentarily lock checkpoint files.
                # After the first attempt, avoid rewriting the meta graph entirely.
                write_meta_graph = False
                time.sleep(min(5.0, 1.0 + attempt))

    def train(self, ds, epochs, retrain=False, test_data=False,
              early_stopping=True, patience=2, min_delta=3.162278, evaluation_split="validation"):
        print('\nTraining...\n')

        training_start_time = time.time()

        # Create log file in the output directory
        log_path = os.path.join(self.out_dir, "training_log.txt")
        log_file = open(log_path, "w")

        def log(msg):
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        model = self.model

        with model.graph.as_default():
            config = tf.ConfigProto(
                device_count={'GPU': len(model.gpus)},
                gpu_options=tf.GPUOptions(allow_growth=True),
                allow_soft_placement=True,
                log_device_placement=False
            )

            # Pararellizing ops (default=2)
            # config.intra_op_parallelism_threads = 5
            # Executing ops in parallel (default=5)
            # config.inter_op_parallelism_threads = 10

            with tf.Session(config=config) as sess:
                restore_path = os.path.join(self.model_dir, 'model.ckpt')
                best_train_ckpt = os.path.join(self.out_dir, 'best.ckpt')
                best_eval_ckpt = os.path.join(self.out_dir, 'best_eval.ckpt')
                final_ckpt = os.path.join(self.out_dir, 'model.ckpt')

                if retrain:
                    model.saver.restore(sess, restore_path)
                    log('Restored existing checkpoint from {}'.format(restore_path))
                else:
                    sess.run(model.init)
                    log('Initialized new model.')

                lowest_train = float('inf')
                best_eval = float('inf')
                wait = 0
                stopped_early = False
                best_epoch = -1
                evaluation_split = str(evaluation_split).strip().lower()
                use_validation = (
                    evaluation_split == "validation"
                    and hasattr(ds, "val_in_batch")
                    and np.size(ds.val_in_batch) > 0
                )
                eval_in_batch = ds.val_in_batch if use_validation else ds.test_in_batch
                eval_out_batch = ds.val_out_batch if use_validation else ds.test_out_batch
                eval_label = "Validation" if use_validation else "Test"

                for epoch in tqdm(range(epochs)):
                    epoch_start_time = time.time()

                    # --------------------
                    # Train
                    # --------------------
                    model.load_ds(sess, ds.train_in_batch, ds.train_out_batch)
                    train_loss = model.run(sess, (model.train, model.loss))
                    train_loss = np.mean(train_loss[1])
                    self.train_all.append(train_loss)

                    # --------------------
                    # Test
                    # --------------------
                    if test_data:
                        model.load_ds(sess, eval_in_batch, eval_out_batch)
                        test_loss = model.run(sess, model.loss)
                        test_loss = np.mean(test_loss)
                        self.test_all.append(test_loss)
                    else:
                        test_loss = float('nan')

                    # --------------------
                    # Timing
                    # --------------------
                    epoch_time = time.time() - epoch_start_time
                    elapsed_time = time.time() - training_start_time

                    completed_epochs = epoch + 1
                    avg_epoch_time = elapsed_time / completed_epochs
                    remaining_epochs = epochs - completed_epochs
                    estimated_remaining_time = avg_epoch_time * remaining_epochs

                    # --------------------
                    # Log current epoch
                    # --------------------
                    log('[{epoch:04}]: Train {train:.6f}, {eval_name} {test:.6f}'.format(
                        epoch=epoch, train=train_loss, test=test_loss
                        , eval_name=eval_label
                    ))
                    log('  -> Epoch time: {:.2f}s | Elapsed: {:.2f} min | Est remaining: {:.2f} min'.format(
                        epoch_time,
                        elapsed_time / 60.0,
                        estimated_remaining_time / 60.0
                    ))

                    # --------------------
                    # Save best training-loss checkpoint
                    # --------------------
                    if train_loss < lowest_train:
                        self._save_checkpoint(model.saver, sess, best_train_ckpt, log)
                        lowest_train = train_loss
                        log('  -> New best training loss. Saved best.ckpt')

                    # --------------------
                    # Early stopping based on test loss
                    # --------------------
                    if early_stopping and test_data:
                        if test_loss < (best_eval - min_delta):
                            best_eval = test_loss
                            best_epoch = epoch
                            wait = 0
                            self._save_checkpoint(model.saver, sess, best_eval_ckpt, log)
                            log('  -> New best {} loss. Saved best_eval.ckpt'.format(eval_label.lower()))
                        else:
                            wait += 1
                            log('  -> No significant {} improvement for {} epoch(s).'.format(
                                eval_label.lower(),
                                wait,
                            ))

                        if wait >= patience:
                            log('')
                            log('Early stopping triggered at epoch {}.'.format(epoch))
                            log('Best {} loss was {:.6f} at epoch {}.'.format(
                                eval_label.lower(),
                                best_eval,
                                best_epoch,
                            ))
                            stopped_early = True
                            break

                # --------------------
                # Save final checkpoint
                # --------------------
                self._save_checkpoint(model.saver, sess, final_ckpt, log)
                log('Saved final checkpoint: model.ckpt')

                # --------------------
                # Final summary
                # --------------------
                total_training_time = time.time() - training_start_time

                if stopped_early:
                    log('Training stopped early.')
                else:
                    log('Training completed all {} epochs.'.format(epochs))

                log('Total training time: {:.2f} seconds ({:.2f} minutes, {:.2f} hours)'.format(
                    total_training_time,
                    total_training_time / 60.0,
                    total_training_time / 3600.0
                ))

        log_file.close()
