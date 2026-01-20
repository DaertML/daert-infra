"""Virtual Machine management module for AutoML using Vagrant."""

import os
import subprocess
import json
from typing import Dict, List, Optional, Any
from pathlib import Path


class VMManager:
    """Manager class for VirtualBox VM operations using Vagrant."""

    def __init__(self, vm_dir: str = "./vms", provider: str = "virtualbox"):
        self.vm_dir = Path(vm_dir)
        self.provider = provider
        self.vagrant_dir = self.vm_dir / "master"

    def _ensure_vagrant_dir(self) -> None:
        """Ensure Vagrant directory exists."""
        self.vagrant_dir.mkdir(parents=True, exist_ok=True)

    def init_vagrant(self, box: str = "ubuntu/focal64",
                     memory: str = "4096", cpus: int = 2,
                     ip: str = "192.168.56.10",
                     hostname: str = "automl-master") -> None:
        """Initialize Vagrant configuration."""
        self._ensure_vagrant_dir()

        vagrantfile_content = f'''Vagrant.configure("2") do |config|
  config.vm.box = "{box}"

  config.vm.hostname = "{hostname}"

  config.vm.network "private_network", ip: "{ip}"

  config.vm.provider "virtualbox" do |vb|
    vb.memory = "{memory}"
    vb.cpus = {cpus}
    vb.name = "{hostname}"
    vb.gui = false
  end

  config.vm.synced_folder "./experiments", "/home/vagrant/experiments",
    create: true, owner: "vagrant", group: "vagrant"

  config.vm.synced_folder "./mlflow_artifacts", "/home/vagrant/mlflow/artifacts",
    create: true, owner: "vagrant", group: "vagrant"

  config.vm.provision "shell", inline: <<-SHELL
    apt-get update
    apt-get install -y docker.io docker-compose
    systemctl enable docker
    systemctl start docker
    usermod -aG docker vagrant
    apt-get install -y python3-pip python3-venv
  SHELL
end
'''

        with open(self.vagrant_dir / "Vagrantfile", 'w') as f:
            f.write(vagrantfile_content)

        (self.vagrant_dir / "experiments").mkdir(exist_ok=True)
        (self.vagrant_dir / "mlflow_artifacts").mkdir(exist_ok=True)

    def _run_vagrant_command(self, command: List[str]) -> subprocess.CompletedProcess:
        """Run a Vagrant command."""
        result = subprocess.run(
            ["vagrant"] + command,
            cwd=self.vagrant_dir,
            capture_output=True,
            text=True
        )
        return result

    def up(self, provision: bool = True) -> bool:
        """Start and provision the VM."""
        cmd = ["up", f"--provider={self.provider}"]
        if not provision:
            cmd.append("--no-provision")

        result = self._run_vagrant_command(cmd)
        return result.returncode == 0

    def halt(self) -> bool:
        """Halt the VM."""
        result = self._run_vagrant_command(["halt"])
        return result.returncode == 0

    def reload(self) -> bool:
        """Reload the VM."""
        result = self._run_vagrant_command(["reload"])
        return result.returncode == 0

    def destroy(self) -> bool:
        """Destroy the VM."""
        result = self._run_vagrant_command(["destroy", "-f"])
        return result.returncode == 0

    def status(self) -> Dict[str, Any]:
        """Get VM status."""
        result = self._run_vagrant_command(["status", "--machine-readable"])

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'running' in line.lower():
                    return {'status': 'running', 'details': line}
                elif 'not created' in line.lower() or 'poweroff' in line.lower():
                    return {'status': 'stopped', 'details': line}

        return {'status': 'unknown', 'details': result.stdout}

    def ssh(self, command: str) -> tuple:
        """Run SSH command on the VM."""
        result = self._run_vagrant_command(["ssh", "-c", command])
        return result.returncode, result.stdout, result.stderr

    def provision_with_ansible(self, playbook_path: str) -> bool:
        """Provision VM with Ansible playbook."""
        result = self._run_vagrant_command([
            "provision",
            "--provision-with",
            "shell"
        ])

        if playbook_path:
            self.ssh(f"mkdir -p /home/vagrant/ansible")
            self.ssh(f"echo '{open(playbook_path).read()}' > /home/vagrant/ansible/playbook.yml")
            self.ssh("cd /home/vagrant/ansible && ansible-playbook playbook.yml")

        return result.returncode == 0

    def get_ip(self) -> str:
        """Get VM IP address."""
        result = self._run_vagrant_command(["ssh", "-c", "hostname -I | awk '{print $1}'"])

        if result.returncode == 0:
            return result.stdout.strip()

        return ""

    def get_ssh_config(self) -> Dict[str, str]:
        """Get SSH configuration for the VM."""
        result = self._run_vagrant_command(["ssh-config"])

        config = {}
        for line in result.stdout.split('\n'):
            if 'HostName' in line:
                config['hostname'] = line.split()[-1]
            elif 'User' in line:
                config['user'] = line.split()[-1]
            elif 'Port' in line:
                config['port'] = line.split()[-1]
            elif 'IdentityFile' in line:
                config['key'] = line.split()[-1]

        return config

    def suspend(self) -> bool:
        """Suspend the VM."""
        result = self._run_vagrant_command(["suspend"])
        return result.returncode == 0

    def resume(self) -> bool:
        """Resume a suspended VM."""
        result = self._run_vagrant_command(["resume"])
        return result.returncode == 0

    def snapshot_take(self, name: str) -> bool:
        """Take a snapshot of the VM."""
        result = self._run_vagrant_command(["snapshot", "take", name])
        return result.returncode == 0

    def snapshot_restore(self, name: str) -> bool:
        """Restore a VM snapshot."""
        result = self._run_vagrant_command(["snapshot", "restore", name])
        return result.returncode == 0

    def snapshot_list(self) -> List[str]:
        """List VM snapshots."""
        result = self._run_vagrant_command(["snapshot", "list"])

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            return [line.strip() for line in lines if line.strip()]

        return []

    def run_docker(self, command: str) -> tuple:
        """Run Docker command on the VM."""
        return self.ssh(f"docker {command}")

    def run_docker_compose(self, command: str, compose_dir: str = "/home/vagrant") -> tuple:
        """Run Docker Compose command on the VM."""
        return self.ssh(f"cd {compose_dir} && docker-compose {command}")

    def copy_to_vm(self, local_path: str, remote_path: str) -> bool:
        """Copy file to VM using scp."""
        config = self.get_ssh_config()

        result = subprocess.run([
            "scp",
            "-i", config.get('key', ''),
            "-o", "StrictHostKeyChecking=no",
            local_path,
            f"{config.get('user', 'vagrant')}@{config.get('hostname', 'localhost')}:{remote_path}"
        ], capture_output=True, text=True)

        return result.returncode == 0

    def copy_from_vm(self, remote_path: str, local_path: str) -> bool:
        """Copy file from VM using scp."""
        config = self.get_ssh_config()

        result = subprocess.run([
            "scp",
            "-i", config.get('key', ''),
            "-o", "StrictHostKeyChecking=no",
            f"{config.get('user', 'vagrant')}@{config.get('hostname', 'localhost')}:{remote_path}",
            local_path
        ], capture_output=True, text=True)

        return result.returncode == 0

    def install_requirements(self, requirements_path: str = "/home/vagrant/requirements.txt") -> tuple:
        """Install Python requirements on VM."""
        return self.ssh(f"pip3 install -r {requirements_path}")

    def setup_mlflow(self) -> tuple:
        """Setup MLflow on VM."""
        commands = [
            "mkdir -p /home/vagrant/mlflow",
            "cd /home/vagrant/mlflow",
            "docker-compose up -d"
        ]
        return self.ssh(" && ".join(commands))

    def create_docker_compose(self) -> None:
        """Create docker-compose.yml on VM."""
        compose_content = '''version: '3.8'

services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v3.8.1
    ports:
      - "5000:5000"
    volumes:
      - ./artifacts:/home/vagrant/mlflow/artifacts
    environment:
      - MLFLOW_TRACKING_URI=http://localhost:5000
      - MLFLOW_ALLOW_HTTP_REQUESTS=true
    command: mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root /home/vagrant/mlflow/artifacts --serve-artifacts
    networks:
      - automl_network
    restart: always

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    networks:
      - automl_network
    restart: always

networks:
  automl_network:
    driver: bridge
'''
        with open(self.vagrant_dir / "docker-compose.yml", 'w') as f:
            f.write(compose_content)

    def get_vm_info(self) -> Dict[str, Any]:
        """Get comprehensive VM information."""
        status = self.status()
        ip = self.get_ip()
        ssh_config = self.get_ssh_config()

        return {
            'status': status.get('status', 'unknown'),
            'ip': ip,
            'hostname': ssh_config.get('hostname', ''),
            'user': ssh_config.get('user', 'vagrant'),
            'port': ssh_config.get('port', '22'),
            'provider': self.provider
        }
