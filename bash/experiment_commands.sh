python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base-deci-hybrid_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=2 trainer.auto_scale_batch_size=True model=base_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=2 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base-deci-hybrid_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=2 trainer.auto_scale_batch_size=True model=base_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=2 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base-deci-hybrid_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=2 trainer.auto_scale_batch_size=True model=base_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=2 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base-deci-hybrid_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=2 trainer.auto_scale_batch_size=True model=base_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=2 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base-deci-hybrid_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=2 trainer.auto_scale_batch_size=True model=base_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=2 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base-deci-hybrid_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=2 trainer.auto_scale_batch_size=True model=base_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=2 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=False datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base-deci-hybrid_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=2 trainer.auto_scale_batch_size=True model=base_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=2 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base-deci-hybrid_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=2 trainer.auto_scale_batch_size=True model=base_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=2 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base-deci-hybrid_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=2 trainer.auto_scale_batch_size=True model=base_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=2 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base-deci-hybrid_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=2 trainer.auto_scale_batch_size=True model=base_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=2 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=False

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base-deci-hybrid_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=2 trainer.auto_scale_batch_size=True model=base_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=2 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base-deci-hybrid_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=2 trainer.auto_scale_batch_size=True model=base_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=2 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=False datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base-deci-hybrid_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=2 trainer.auto_scale_batch_size=True model=base_modus_prime_resnet50 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=2 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base_modus_prime_vi-transformer16 datamodule=base-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base-deci-hybrid_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=10 trainer.gpus=2 trainer.auto_scale_batch_size=True model=base_modus_prime_resnet50 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=16 trainer.gpus=1 trainer.auto_scale_batch_size=True model=milli_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=8 trainer.gpus=1 trainer.auto_scale_batch_size=True model=centi_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=4 trainer.gpus=1 trainer.auto_scale_batch_size=True model=deci_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

python run.py hydra.verbose=True trainer=default resume=True batch_size=2 trainer.gpus=1 trainer.auto_scale_batch_size=True model=base_modus_prime_vi-transformer16 datamodule=hecta-tali datamodule.config.modality_config.image=True datamodule.config.modality_config.text=True datamodule.config.modality_config.audio=True datamodule.config.modality_config.video=True

