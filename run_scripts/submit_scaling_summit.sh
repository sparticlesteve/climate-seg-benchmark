bsub_args="-P stf011 -J climseg-summit -alloc_flags nvme  -o climseg-summit.o%J -W 0:45"

for i in $(seq 0 8); do 
	if (( i < 8 )); then 
		nodes=$((2**i))
	else
		nodes=170 # cap total ranks below 1024
	fi 
      	ntrain=$((nodes*6*8))
	echo bsub -nnodes $nodes $bsub_args "./train_summit.sh --ntrain $ntrain --nvalid 0 --epochs 4"
	bsub -nnodes $nodes $bsub_args "./train_summit.sh --ntrain $ntrain --nvalid 0 --epochs 4"
done

