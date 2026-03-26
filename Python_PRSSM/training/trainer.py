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

    def _save_checkpoint(self, saver, sess, checkpoint_path, log, retries=3):
        meta_path = checkpoint_path + '.meta'
        write_meta_graph = not os.path.exists(meta_path)

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
                time.sleep(1.0)

    def _summarize_training_diagnosis(self, patience, stopped_early):
        if not self.train_all or not self.test_all:
            return {
                'status': 'unavailable',
                'reason': 'Test-loss history is not available, so overtraining cannot be evaluated.',
            }

        train_hist = np.asarray(self.train_all, dtype=float)
        test_hist = np.asarray(self.test_all, dtype=float)

        best_epoch = int(np.argmin(test_hist))
        best_test = float(test_hist[best_epoch])
        final_test = float(test_hist[-1])
        initial_test = float(test_hist[0])
        train_at_best = float(train_hist[best_epoch])
        final_train = float(train_hist[-1])
        epochs_after_best = int(len(test_hist) - 1 - best_epoch)

        denom_best = max(abs(best_test), 1e-12)
        denom_initial = max(abs(initial_test), 1e-12)
        denom_train_best = max(abs(train_at_best), 1e-12)

        test_degradation_from_best = (final_test - best_test) / denom_best
        best_test_improvement = (initial_test - best_test) / denom_initial
        train_improvement_after_best = (train_at_best - final_train) / denom_train_best

        recent_window = min(5, len(test_hist))
        recent_test_change = 0.0
        recent_train_change = 0.0
        if recent_window >= 2:
            recent_test_change = float(test_hist[-recent_window] - test_hist[-1])
            recent_train_change = float(train_hist[-recent_window] - train_hist[-1])

        if (
            epochs_after_best >= max(3, patience)
            and test_degradation_from_best >= 0.02
            and train_improvement_after_best >= 0.01
        ):
            return {
                'status': 'overtraining',
                'reason': (
                    'Test loss reached its minimum at epoch {} and then rose by {:.2%}, '
                    'while training loss still improved by {:.2%}.'
                ).format(best_epoch, test_degradation_from_best, train_improvement_after_best),
            }

        if (
            not stopped_early
            and best_epoch >= len(test_hist) - recent_window
            and recent_test_change > 0.0
            and recent_train_change > 0.0
            and best_test_improvement >= 0.01
        ):
            return {
                'status': 'undertraining',
                'reason': (
                    'Training ended while both train and test loss were still improving; '
                    'the best test loss occurred near the final epoch (epoch {}).'
                ).format(best_epoch),
            }

        if stopped_early and epochs_after_best <= max(1, patience):
            return {
                'status': 'well-balanced',
                'reason': (
                    'Early stopping triggered close to the best test epoch, '
                    'which suggests the stopping point was reasonably timed.'
                ),
            }

        return {
            'status': 'inconclusive',
            'reason': (
                'Train/test loss trends do not strongly indicate overtraining or undertraining '
                'based on the current heuristic.'
            ),
        }

    def train(self, ds, epochs, retrain=False, test_data=False,
              early_stopping=True, patience=3, min_delta=100000):
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

                if retrain:
                    model.saver.restore(sess, self.model_dir + 'model.ckpt')
                    log('Restored existing checkpoint from {}model.ckpt'.format(self.model_dir))
                else:
                    sess.run(model.init)
                    log('Initialized new model.')

                lowest_train = float('inf')
                best_test = float('inf')
                wait = 0
                stopped_early = False
                best_epoch = -1

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
                        model.load_ds(sess, ds.test_in_batch, ds.test_out_batch)
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
                    log('[{epoch:04}]: Train {train:.6f}, Test {test:.6f}'.format(
                        epoch=epoch, train=train_loss, test=test_loss
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
                        self._save_checkpoint(model.saver, sess, self.out_dir + '/best.ckpt', log)
                        lowest_train = train_loss
                        log('  -> New best training loss. Saved best.ckpt')

                    # --------------------
                    # Early stopping based on test loss
                    # --------------------
                    if early_stopping and test_data:
                        if test_loss < (best_test - min_delta):
                            best_test = test_loss
                            best_epoch = epoch
                            wait = 0
                            self._save_checkpoint(model.saver, sess, self.out_dir + '/best_test.ckpt', log)
                            log('  -> New best test loss. Saved best_test.ckpt')
                        else:
                            wait += 1
                            log('  -> No significant test improvement for {} epoch(s).'.format(wait))

                        if wait >= patience:
                            log('')
                            log('Early stopping triggered at epoch {}.'.format(epoch))
                            log('Best test loss was {:.6f} at epoch {}.'.format(best_test, best_epoch))
                            stopped_early = True
                            break

                # --------------------
                # Save final checkpoint
                # --------------------
                self._save_checkpoint(model.saver, sess, self.out_dir + '/model.ckpt', log)
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

                diagnosis = self._summarize_training_diagnosis(
                    patience=patience,
                    stopped_early=stopped_early,
                )
                log('Training diagnosis: {}'.format(diagnosis['status']))
                log('  -> {}'.format(diagnosis['reason']))

        log_file.close()
