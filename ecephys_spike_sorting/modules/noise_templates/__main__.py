from argschema import ArgSchemaParser
import os
import logging
import time

import numpy as np

from .id_noise_templates import id_noise_templates, id_noise_templates_rf

from ...common.utils import write_cluster_group_tsv, load_kilosort_data


def classify_noise_templates(args):

    print('ecephys spike sorting: noise templates module')
    
    start = time.time()

    spike_times, spike_clusters, spike_templates, amplitudes, templates, channel_map, \
    channel_pos, cluster_ids, cluster_quality, cluster_amplitude = \
            load_kilosort_data(args['directories']['kilosort_output_directory'], \
                args['ephys_params']['sample_rate'], \
                convert_to_seconds = True)

    cluster_ids_orig = cluster_ids
    if args['noise_waveform_params']['use_random_forest']:
        # use random forest classifier
        cluster_ids, is_noise = id_noise_templates_rf(spike_times, spike_clusters, \
                    cluster_ids, templates, args['noise_waveform_params'])
    else:
        # use heuristics to identify templates that look like noise
        cluster_ids, is_noise = id_noise_templates(cluster_ids, templates, np.squeeze(channel_map), \
            args['noise_waveform_params'])

    # mapping = {False: 'unsorted', True: 'noise'}
    # labels = [mapping[value] for value in is_noise]
    assert len(cluster_ids_orig) == len(cluster_ids)
    assert np.all(cluster_ids_orig == cluster_ids)  # Sanity check
    assert len(is_noise) == len(cluster_ids)
    labels = cluster_quality  # Keep original labels and only modify noise units
    for idx in np.where(is_noise)[0]:
        labels[cluster_ids[idx]] = 'noise'


    write_cluster_group_tsv(cluster_ids, 
                            labels, 
                            args['directories']['kilosort_output_directory'], 
                            args['ephys_params']['cluster_group_file_name'])
    
    execution_time = time.time() - start

    print('total time: ' + str(np.around(execution_time,2)) + ' seconds')
    print()
    
    return {"execution_time" : execution_time} # output manifest


def main():

    from ._schemas import InputParameters, OutputParameters

    mod = ArgSchemaParser(schema_type=InputParameters,
                          output_schema_type=OutputParameters)

    output = classify_noise_templates(mod.args)

    output.update({"input_parameters": mod.args})
    if "output_json" in mod.args:
        mod.output(output, indent=2)
    else:
        print(mod.get_output_json(output))


if __name__ == "__main__":
    main()
