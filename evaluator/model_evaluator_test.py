from __future__ import print_function
import torch
import optim
import util
from evaluator.average_meter import AverageMeter
from openpyxl import load_workbook
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelEvaluator_test(object):
    """Class for evaluating a model during training"""
    def __init__(self, args, data_loaders, epochs_per_eval=1):
        """
        Args:
            data_loaders: List of Torch 'DataLoader's to sample from.
            num_visuals: Number of visuals to display from the validation set.
            max_eval: Maximum number of examples to evaluate at each evaluation.
            epochs_per_eval: Number of epochs between each evaluations.
        """
        self.args = args
        self.dataset = args.dataset
        self.data_loaders = data_loaders
        self.epochs_per_eval = epochs_per_eval
        self.loss_fn = optim.get_loss_fn(args.loss_fn, args)
        self.name = args.name
        self.lam = args.lam
        self.pred_type = args.pred_type
        self.model_dist = args.model_dist
        self.num_cat_bins = args.num_cat_bins
        self.loss_fn_name = args.loss_fn
        self.num_xcal_bins = args.num_xcal_bins
        if self.model_dist in ['cat', 'mtlr']:
            self.mid_points = args.mid_points
            self.bin_boundaries = args.bin_boundaries

    def evaluate(self, model, device, epoch=None):
        """
        Evaluate a model at the end of the given epoch.

        Args:
            model: Model to evaluate.
            device: Device on which to evaluate the model.
            epoch: The epoch that just finished. Determines whether to evaluate the model.

        Returns:
            metrics: Dictionary of metrics for the current model.

        Notes:
            Returned dictionary will be empty if not an evaluation epoch.
        """
        metrics = {}

        if epoch is None or epoch % self.epochs_per_eval == 0:
            # Evaluate on the training and validation sets
            model.eval()
            for data_loader in self.data_loaders:
                phase_metrics = self._eval_phase(model, data_loader, data_loader.phase, device)
                metrics.update(phase_metrics)
            model.train()

        return metrics

    def _eval_phase(self, model, data_loader, phase, device):
        print("CURRENT EVAL PHASE IS ", phase)
        """Evaluate a model for a single phase.

        Args:
            model: Model to evaluate.
            data_loader: Torch DataLoader to sample from.
            phase: Phase being evaluated. One of 'train', 'val', or 'test'.
            device: Device on which to evaluate the model.

        Returns:
            metrics: Dictionary of metrics for the phase.
        """
        # Keep track of task-specific records needed for computing overall metrics
        records = {'loss_meter': AverageMeter()}
        
        num_examples = len(data_loader.dataset)
        # Sample from the data loader and record model outputs
        loss_fn = self.loss_fn
        num_evaluated = 0

        is_dead_per_batch = []

        all_cdf = []
        all_tte = []
        for src, tgt in data_loader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tte = tgt[:, 0]

            if self.model_dist in ['cat', 'mtlr']:
                tgt = util.cat_bin_target(self.args, tgt, self.bin_boundaries) # check again
                
            # THIS MUST COME AFTER THE ABOCE CAT BIN TARGET FUNCTION
            # BECAUSE CAT BIN TARGET CAN CHANGE SOME IS_DEAD
            is_dead = tgt[:, 1]
            if self.model_dist == 'cox':
                order = torch.argsort(tte)
                is_dead = is_dead[order]
                tte = tte[order]
                
            is_dead_per_batch.append(is_dead)

            if num_evaluated >= num_examples:
                break

            pred_params = model.forward(src)
            cdf = util.get_cdf_val(pred_params, tgt, self.args) # check again
            all_cdf.append(cdf)
            loss = loss_fn(pred_params, tgt, model_dist=self.model_dist)
            num_both = pred_params.size()[0]
            all_tte.append(tte)

            self._record_batch(num_both, loss, **records)
            num_evaluated += src.size(0)
            
        concordance = util.concordance(self.args, data_loader, model)
        
        is_dead = torch.cat(is_dead_per_batch).long()
        
        all_cdf = torch.cat(all_cdf)

        all_tte = torch.cat(all_tte)
        
        if self.args.model_dist == 'mtlr':
            weight = model.get_weight()
            regularizer = util.ridge_norm(weight)*self.args.C1/2 + util.fused_norm(weight)*self.args.C2/2
        
        # Map to summary dictionaries
        metrics = self._get_summary_dict(phase, **records)
        approx_s_calibration = util.s_calibration(points=all_cdf, phase=phase, is_dead=is_dead, args=self.args, gamma=1e5, differentiable=False, device=DEVICE)
        KS_cal, KS = util.get_p_value(args=self.args, cdf=all_cdf, is_dead=is_dead, device=DEVICE) # check again
        approx_d_calibration_20 = util.d_calibration(points=all_cdf, is_dead=is_dead, args=self.args, phase=phase, nbins=self.num_xcal_bins, gamma=1e5, differentiable=False, device=DEVICE)
        
        metrics[phase + '_' + 'NLL'] = metrics[phase + '_' + 'loss']
        metrics[phase + '_' + 'concordance'] = concordance
        metrics[phase + '_' + 'S-cal(20)'] = approx_s_calibration
        metrics[phase + '_' + 'D-cal(20)'] = approx_d_calibration_20
        
        if self.model_dist == 'mtlr':
            metrics[phase + '_' + 'loss'] = metrics[phase + '_' + 'loss'] + self.lam * KS_cal + regularizer
        else:
            metrics[phase + '_' + 'loss'] = metrics[phase + '_' + 'loss'] + self.lam * KS_cal
        
        metrics[phase + '_' + 'KS-cal'] = KS_cal
        metrics[phase + '_' + 'KS'] = KS
        
        if phase == 'train':
            train_tte = all_tte
            train_event = is_dead
            torch.save(train_tte, "./train_tte.pt")
            torch.save(train_event, "./train_event.pt")

        if phase == 'test':
            train_tte = torch.load("./train_tte.pt", weights_only=True)
            train_event = torch.load("./train_event.pt", weights_only=True)
            import os
            if os.path.exists("./train_tte.pt"):
                os.remove("./train_tte.pt")
            if os.path.exists("./train_event.pt"):
                os.remove("./train_event.pt")
                
            cdf_km, tte_km, is_dead_km = util.cdf_all_points(args=self.args)
            km_calibration = util.km_calibration(cdf=cdf_km, tte=tte_km, is_dead=is_dead_km, device=DEVICE)
            test_tte = tte_km
            test_event = is_dead_km
            IBS = util.integrated_brier_score(train_tte=train_tte, train_event=train_event, 
                                            test_tte=test_tte, test_event=test_event, 
                                            cdf_test=cdf_km, time=test_tte)
            metrics[phase + '_' + 'KM-cal'] = km_calibration
            metrics[phase + '_' + 'IBS'] = IBS

        print(' ---- {} epoch Concordance {:.4f}'.format(phase, concordance))
        print(' ---- {} epoch end S-cal(20) {:.5f}'.format(phase, approx_s_calibration))
        print(' ---- {} epoch end D-cal(20) {:.5f}'.format(phase, approx_d_calibration_20))
        print(' ---- {} epoch end KS-cal {:.5f}'.format(phase, KS))

        return metrics

    @staticmethod
    def _record_batch(N, loss, loss_meter=None):
        """
        Record results from a batch to keep track of metrics during evaluation.

        Args:
            logits: Batch of logits output by the model.
            loss_meter: AverageMeter keeping track of average loss during evaluation.
        """
        if loss_meter is not None:
            loss_meter.update(loss.item(), N)

    @staticmethod
    def _get_summary_dict(phase, loss_meter=None):
        """
        Get summary dictionaries given dictionary of records kept during evaluation.

        Args:
            phase: Phase being evaluated. One of 'train', 'val', or 'test'.
            loss_meter: AverageMeter keeping track of average loss during evaluation.

        Returns:
            metrics: Dictionary of metrics for the current model.
        """
        metrics = {phase + '_' + 'loss': loss_meter.avg}
        
        return metrics

    @staticmethod
    def _write_summary_stats(phase, loss_meter=None):
        """
        Write stats of evaluation to file.

        Args:
            phase: Phase being evaluated. One of 'train', 'val', or 'test'.
            loss_meter: AverageMeter keeping track of average loss during evaluation.

        Returns:
            metrics: Dictionary of metrics for the current model.
        """
        metrics = {phase + '_' + 'loss': loss_meter.avg}

        return metrics


