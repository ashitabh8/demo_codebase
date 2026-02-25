import torch
import numpy as np
import sys
import os

def fft_preprocess(time_loc_inputs):
    """Run FFT on the time-domain input.
    time_loc_inputs: [b, c, i, s]
    freq_loc_inputs: [b, c, i, s]
    """
    freq_loc_inputs = dict()

    for loc in time_loc_inputs:
        freq_loc_inputs[loc] = dict()
        for mod in time_loc_inputs[loc]:
            loc_mod_freq_output = torch.fft.fft(time_loc_inputs[loc][mod], dim=-1)
            loc_mod_freq_output = torch.view_as_real(loc_mod_freq_output)
            loc_mod_freq_output = loc_mod_freq_output.permute(0, 1, 4, 2, 3)
            b, c1, c2, i, s = loc_mod_freq_output.shape
            loc_mod_freq_output = loc_mod_freq_output.reshape(b, c1 * c2, i, s)
            freq_loc_inputs[loc][mod] = loc_mod_freq_output

    return freq_loc_inputs

def load_data_index(data_index):
    stage = ['train', 'val', 'test']
    index_files = []
    for s in stage:
        data_index_path = os.path.join(data_index, f"{s}_index.txt")
        index_file = np.loadtxt(data_index_path, dtype=str)    
        if len(index_file) == 0:
            continue
        index_files.append(index_file)

    index_files = np.concatenate(index_files)
    index_files = list(set(list(index_files)))
    return index_files

def compute_psd(data_fft):
    energy_spectrum = dict()
    for loc in data_fft:
        energy_spectrum[loc] = dict()
        for mod in data_fft[loc]:
            energy_spectrum[loc][mod] = torch.abs(data_fft[loc][mod]) ** 2
    return energy_spectrum

def compute_energy_spectrum(samples):
    # scale energy spectrum for energy distribution
    min_psd = {}
    max_psd = {}
    for sample in samples.values():
        for loc in sample['psd']:
            if loc not in min_psd:
                min_psd[loc] = dict()
                max_psd[loc] = dict()
            for mod in sample['psd'][loc]:
                if mod not in min_psd[loc]:
                    min_psd[loc][mod] = sample['psd'][loc][mod]
                    max_psd[loc][mod] = sample['psd'][loc][mod]
                else:
                    min_psd[loc][mod] = torch.minimum(min_psd[loc][mod], sample['psd'][loc][mod])
                    max_psd[loc][mod] = torch.maximum(max_psd[loc][mod], sample['psd'][loc][mod])
    
    for sample in samples.values():
        sample['energy_spectrum'] = {}
        for loc in sample['psd']:
            if loc not in sample['energy_spectrum']:
                sample['energy_spectrum'][loc] = {}
            for mod in sample['psd'][loc]:
                sample['energy_spectrum'][loc][mod] = (sample['psd'][loc][mod] - min_psd[loc][mod]) / (max_psd[loc][mod] - min_psd[loc][mod] + 1e-8)
    
    return samples

def compute_coherence(samples):
    # Compute average coherence for coherence distribution
    for sample_id, sample in samples.items():
        sample['coherence'] = {}
        for loc in sample['fft']:
            sample['coherence'][loc] = {}
            for mod in sample['fft'][loc]:
                
                # initialize coherence sum for sample i
                coherence_sum = np.zeros_like(sample['fft'][loc][mod], dtype=np.float64)
                count = 0
                for other_id, other_sample in samples.items():
                    if sample_id == other_id:
                        continue # D - 1
                        
                    # compute cross-PSD
                    cross_psd = sample['fft'][loc][mod] * torch.conj(other_sample['fft'][loc][mod])
                    psd_product = np.multiply(np.abs(sample['fft'][loc][mod]) ** 2, np.abs(other_sample['fft'][loc][mod]) ** 2) + 1e-8
                    
                    # compute cross coherence and add to coherence sum
                    coherence_sum = np.add(coherence_sum, (np.abs(cross_psd) ** 2 / psd_product).real)
                    count += 1
                
                # sample i average coherence
                sample['coherence'][loc][mod] = coherence_sum / count if count > 0 else coherence_sum

    return samples

def set_energy_coherence_for_dataset(data_index):
    index_files = load_data_index(data_index)
    samples = {}
    
    
    print(f"Compute PSD for {len(index_files)} samples")
    # compute Power Spectrum Density (PSD)
    for pt_path in index_files:
        if pt_path not in samples:
            samples[pt_path] = {}
        
        sample = torch.load(pt_path, weights_only=False)
        data = sample['data']
        for loc in data:
            for mod in data[loc]:
                data[loc][mod] = data[loc][mod].unsqueeze(0)
        
        # print_data_shape(data)
        data_fft = fft_preprocess(data)
        
        psd = compute_psd(data_fft)
        samples[pt_path]['psd'] = psd
        samples[pt_path]['fft'] = data_fft        
    
    print(f"Compute energy spectrum for {len(samples)} samples")
    samples = compute_energy_spectrum(samples)
    
    print(f"Compute coherence for {len(samples)} samples")
    samples = compute_coherence(samples)
    
    return samples

def save_samples(samples):
    print(f"Save samples for {len(samples)} samples")
    for sample_pt in samples:
        print(samples[sample_pt].keys())
        
        original_sample = torch.load(sample_pt, weights_only=False)
        original_sample['energy_spectrum'] = samples[sample_pt]['energy_spectrum']
        original_sample['coherence'] = samples[sample_pt]['coherence']
        torch.save(original_sample, sample_pt)
        


if __name__ == "__main__":
    data_index = sys.argv[1]
    if not os.path.exists(data_index):
        raise FileNotFoundError(f"Data index file {data_index} not found")
    
    print(f"Set energy and coherence for {data_index}")
    samples = set_energy_coherence_for_dataset(data_index)
    save_samples(samples)