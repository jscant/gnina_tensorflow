"""
Sneaky daemon script for restarting OOM autoencoders.

Usage: put on desktop and submit as slurm job, with low memory (~100 MB), 1 CPU
and no GPUs. It will restart itself once per day to get around the one day
maximum run time limit on the naga-small queue.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

# Slurm template
template = """#!/bin/bash                                               
#SBATCH -J {0} # TITLE_TEMPALTE
#SBATCH -A opig                         # Project Account                                           
#SBATCH --time=3-00:00:00                 # Walltime                                                  
#SBATCH --mem-per-cpu=64000            # memory/cpu (in MB) ### commented out                      
#SBATCH --ntasks=1                      # 1 tasks                                                   
#SBATCH --cpus-per-task=1               # number of cores per task                                 
#SBATCH --nodes=1                       # number of nodes                                           
#SBATCH --nodelist {2}
#SBATCH --chdir=/data/localhost/not-backed-up/scantleb
#SBATCH --partition=naga-gpu-64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/data/localhost/not-backed-up/scantleb/slurm_logs/slurm_%j.out                             
#SBATCH --error=/data/localhost/not-backed-up/scantleb/slurm_logs/slurm_%j.out

cd ~/gnina_tensorflow
python3 autoencoder/gnina_autoencoder.py {1} --resume
"""


def get_job_ids():
    """Gives a set of tuples with running job ids and job names."""
    output = subprocess.run(
        ['sq'], capture_output=True, check=True, shell=True).stdout
    output = output.decode()[:-1]
    jobs = [entry for entry in [line.split() for line in output.split('\n')]]
    jobs = [(job[0], job[2]) for job in jobs if job[2] not in ['bash', 'daemon']
            and not job[0].startswith('NM') and job[-1].find('02') != -1]
    jobs = set([(job[0], job[1]) for job in jobs])
    return jobs


def get_status(job_id):
    """Get the status (running, finished, OOM, cancelled, pending) for a job."""
    sacct = subprocess.run(
        ['sacct', '-j', job_id], capture_output=True, check=True,
        shell=False)
    status = sacct.stdout.decode().split('\n')[-2].split()[-2]
    return status.lower()


def get_working_dir(job_id):
    """Get the working directory for an autoencoder slurm job."""
    with open(Path('~/slurm_logs/slurm_{}.out'.format(
            job_id)).expanduser(), 'r') as f:
        for line in f.readlines():
            if line.startswith('absolute_save_path'):
                return line.split()[-1]
    return None


class JobList:
    """Class for live handling of active, completed and failed slurm jobs."""

    def __init__(self, slurm_node, log_file=None):
        self._statuses = {}
        self._template = template
        self._slurm_node = slurm_node
        if log_file is None:
            self.log_file = None
        else:
            self.log_file = Path(log_file).expanduser().resolve()
        self.log_output('Initialising job handler...')
        for job_id, job_name in get_job_ids():
            status = get_status(job_id)
            self._statuses[job_id] = (status, job_name)
            if self._statuses[job_id][0] not in ['running', 'pending']:
                raise RuntimeError(
                    'Unexpected status: {0} for job id: {1}'.format(
                        status, job_id))
            else:
                self.log_output(
                    'Job {0} ({1}) has started with status {2}.'.format(
                        job_id, job_name, status))

    @property
    def status(self):
        """Return a dictionary with {job_id : (job_status, job_name) mapping."""
        return self._statuses

    def delete_job(self, job_id):
        """Remove a job with the given job id from record of running jobs."""
        del self._statuses[job_id]

    def update_status(self):
        """Check currently and previously running jobs for a change in status.

        Queries Slurm to check for any newly submitted jobs, as well as checking
        whether any jobs previously recorded as running or pending have
        finished, failed or been cancelled.

        If autoencoder jobs with saved model checkpoints are found to have
        failed due to an OOM error, they are restarted from the latest found
        checkpoint.
        """
        failed = set()
        for job_id, (old_status, job_name) in self.status.items():
            new_status = get_status(job_id)
            if new_status != old_status:
                if new_status.startswith('out_of_me'):
                    # OOM failure, pass to handle_failures
                    failed.add((job_id, job_name))
                    self.log_output('Job with id {} has failed. Attempting to '
                                    'restart.'.format(job_id))
                elif new_status.startswith('finished'):
                    # Job has finished so can be safely forgotten about
                    self.delete_job(job_id)
                    self.log_output(
                        'Job with id {} has finished successfully.'.format(
                            job_id))
                elif not new_status.startswith('running'):
                    # Either a manual calculation or something else has gone
                    # wrong
                    self.log_output(
                        'Unexpected status: {0} for job id: {1}'.format(
                            self._statuses[job_id][0], job_id))

        for job_id, job_name in get_job_ids():
            if job_id in self.status.keys() or job_id in [i[0] for i in failed]:
                continue
            # New job: need to determine and record its status
            status = get_status(job_id)
            if status[:7] not in ['running', 'pending']:
                self.log_output('Unexpected status: {0} for job id: {1}'.format(
                    self._statuses[job_id][0], job_id))
            else:
                self.log_output('Found new job with id {0} ({1}).'.format(
                    job_id, job_name))
            self._statuses[job_id] = (status, job_name)

        # Bookkeeping complete; handle any OOM jobs
        self.handle_failures(failed)

    def handle_failures(self, failed):
        """Attempts to restart failed jobs from last saved checkpoints.

        Arguments:
            failed: iterable of (job_id, job_name) iterables denoting details of
                all failed jobs
        """
        for job_id, job_name in failed:
            working_dir = get_working_dir(job_id)
            if working_dir is None:  # Not an autoencoder
                self.log_output('Job {} is not an autoencoder.'.format(job_id))
                continue
            checkpoints_dir = Path(working_dir, 'checkpoints')
            max_iters = -1
            for checkpoint in checkpoints_dir.glob('ckpt_model_*'):
                if not checkpoint.is_dir():
                    continue
                max_iters = max(max_iters, int(str(checkpoint).split('_')[-1]))
            if max_iters == -1:
                # No saved checkpoint; abort this job
                self.log_output('No checkpoint found for {0} ({1}), cannot '
                                'restart.'.format(job_id, job_name))
                self.delete_job(job_id)
                continue
            latest_ckpt = checkpoints_dir / 'ckpt_model_{}'.format(max_iters)
            resubmitted_job = self._template.format(
                job_name, latest_ckpt, self._slurm_node)
            with open(Path('~/_resubmitted_job.sh').expanduser(), 'w') as f:
                f.write(resubmitted_job)
            sbatch = subprocess.run(
                ['sbatch', str(Path('~/_resubmitted_job.sh').expanduser())],
                capture_output=True, check=True,
                shell=False).stdout.decode()[:-1]
            submitted_job_id = sbatch.split('\n')[-1].strip().split()[-1]
            time.sleep(10)
            submitted_job_status = get_status(submitted_job_id)
            if submitted_job_status[:7] not in ['running', 'pending']:
                self.log_output('Unexpected status: {0} for job id: {1}'.format(
                    submitted_job_status, submitted_job_id))
            else:
                self.log_output(
                    'Job with id {0} ({1}) has been sucessfully restarted, '
                    'with new job id {2}.'.format(
                        job_id, job_name, submitted_job_id))
            self.delete_job(job_id)

    def log_output(self, *args, **kwargs):
        """Print to console, and optionally to log file.

        Prints output to log file if specified in constructor of class.

        Arguments:
            *args: items to be printed
            **kwargs: keyworkd arguments for print statement
        """
        print(*args, **kwargs)
        if self.log_file is not None:
            original_stdout = sys.stdout
            with open(self.log_file, 'a') as f:
                sys.stdout = f
                print(*args, **kwargs)
                sys.stdout = original_stdout


if __name__ == '__main__':
    start_time = time.time()
    jl = JobList(os.environ.get('SLURMD_NODENAME'), log_file='~/daemon.log')
    while True:
        jl.update_status()
        time.sleep(1)
        if time.time() - start_time > 80000:  # Just less than a day
            # Restart this process in slurm.
            sbatch = subprocess.run(
                ['sbatch', str(Path('~/_daemon.sh').expanduser())], shell=False)
            exit(0)
