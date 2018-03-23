#!/usr/bin/python 

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import sys
sys.path.append('../')
from Hierarchical_Neural_Networks import LocalSensorNetwork, CloudNetwork, Network
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gzip;
import StringIO;

def parse_header_of_csv(csv_str):
    # Isolate the headline columns:
    headline = csv_str[:csv_str.index('\n')];
    columns = headline.split(',');

    # The first column should be timestamp:
    assert columns[0] == 'timestamp';
    # The last column should be label_source:
    assert columns[-1] == 'label_source';
    
    # Search for the column of the first label:
    for (ci,col) in enumerate(columns):
        if col.startswith('label:'):
            first_label_ind = ci;
            break;
        pass;

    # Feature columns come after timestamp and before the labels:
    feature_names = columns[1:first_label_ind];
    # Then come the labels, till the one-before-last column:
    label_names = columns[first_label_ind:-1];
    for (li,label) in enumerate(label_names):
        # In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:
        assert label.startswith('label:');
        label_names[li] = label.replace('label:','');
        pass;
    
    return (feature_names,label_names);

def parse_body_of_csv(csv_str,n_features):
    # Read the entire CSV body into a single numeric matrix:
    full_table = np.loadtxt(StringIO.StringIO(csv_str),delimiter=',',skiprows=1);
    
    # Timestamp is the primary key for the records (examples):
    timestamps = full_table[:,0].astype(int);
    
    # Read the sensor features:
    X = full_table[:,1:(n_features+1)];
    
    # Read the binary label values, and the 'missing label' indicators:
    trinary_labels_mat = full_table[:,(n_features+1):-1]; # This should have values of either 0., 1. or NaN
    M = np.isnan(trinary_labels_mat); # M is the missing label matrix
    Y = np.where(M,0,trinary_labels_mat) > 0.; # Y is the label matrix
    
    return (X,Y,M,timestamps);

def read_user_data(uuid):
    user_data_file = '%s.features_labels.csv.gz' % uuid;

    # Read the entire csv file of the user:
    with gzip.open(user_data_file,'rb') as fid:
        csv_str = fid.read();
        pass;

    (feature_names,label_names) = parse_header_of_csv(csv_str);
    n_features = len(feature_names);
    (X,Y,M,timestamps) = parse_body_of_csv(csv_str,n_features);

    return (X,Y,M,timestamps,feature_names,label_names);


def project_features_to_selected_sensors(X,feat_sensor_names,sensors_to_use):
    use_feature = np.zeros(len(feat_sensor_names),dtype=bool);
    for sensor in sensors_to_use:
        is_from_sensor = (feat_sensor_names == sensor);
        use_feature = np.logical_or(use_feature,is_from_sensor);
        pass;
    X = X[:,use_feature];
    return X;


def get_sensor_names_from_features(feature_names):
    feat_sensor_names = np.array([None for feat in feature_names]);
    for (fi,feat) in enumerate(feature_names):
        if feat.startswith('raw_acc'):
            feat_sensor_names[fi] = 'Acc';
            pass;
        elif feat.startswith('proc_gyro'):
            feat_sensor_names[fi] = 'Gyro';
            pass;
        elif feat.startswith('raw_magnet'):
            feat_sensor_names[fi] = 'Magnet';
            pass;
        elif feat.startswith('watch_acceleration'):
            feat_sensor_names[fi] = 'WAcc';
            pass;
        elif feat.startswith('watch_heading'):
            feat_sensor_names[fi] = 'Compass';
            pass;
        elif feat.startswith('location'):
            feat_sensor_names[fi] = 'Loc';
            pass;
        elif feat.startswith('location_quick_features'):
            feat_sensor_names[fi] = 'Loc';
            pass;
        elif feat.startswith('audio_naive'):
            feat_sensor_names[fi] = 'Aud';
            pass;
        elif feat.startswith('audio_properties'):
            feat_sensor_names[fi] = 'AP';
            pass;
        elif feat.startswith('discrete'):
            feat_sensor_names[fi] = 'PS';
            pass;
        elif feat.startswith('lf_measurements'):
            feat_sensor_names[fi] = 'LF';
            pass;
        else:
            raise ValueError("!!! Unsupported feature name: %s" % feat);

        pass;

    return feat_sensor_names;    


def estimate_standardization_params(X_train):
    mean_vec = np.nanmean(X_train,axis=0);
    std_vec = np.nanstd(X_train,axis=0);
    return (mean_vec,std_vec);

def standardize_features(X,mean_vec,std_vec):
    # Subtract the mean, to centralize all features around zero:
    X_centralized = X - mean_vec.reshape((1,-1));
    # Divide by the standard deviation, to get unit-variance for all features:
    # * Avoid dividing by zero, in case some feature had estimate of zero variance
    normalizers = np.where(std_vec > 0., std_vec, 1.).reshape((1,-1));
    X_standard = X_centralized / normalizers;
    return X_standard;

def project_features_to_selected_sensors(X,feat_sensor_names,sensors_to_use):
    use_feature = np.zeros(len(feat_sensor_names),dtype=bool);
    for sensor in sensors_to_use:
        is_from_sensor = (feat_sensor_names == sensor);
        use_feature = np.logical_or(use_feature,is_from_sensor);
        pass;
    X = X[:,use_feature];
    return X;

def main(train_data, train_phi, train_label, train_M, test_data, test_phi, test_label, test_M):
    tf.reset_default_graph()	
    train_accuracy_batches = []

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 175])
        w_ = tf.placeholder(tf.float32, [None, 51])
        weighting = tf.placeholder(tf.bool, [None, 51])
        y_ = tf.placeholder(tf.float32, [None, 51])

    # ACC: 26  
    # Gyro: 26
    # WAcc: 46
    # Loc: 17
    # Aud: 26
    # PS: 34

    Acc = LocalSensorNetwork("Acc", x[:,0:26], [256,10])
    Gyro = LocalSensorNetwork("Gyro", x[:,26:52], [256,10])
    WAcc = LocalSensorNetwork("WAcc", x[:,52:98], [256,10])
    Loc = LocalSensorNetwork("Loc", x[:,98:115], [256,10])
    Aud = LocalSensorNetwork("Aud", x[:,115:141], [256,10])
    PS = LocalSensorNetwork("PS", x[:,141:], [256,10])


    cloud = CloudNetwork("cloud", [256,51])
    model = cloud.connect([Acc, Gyro, WAcc, Loc, Aud, PS])

    training_epochs = 20
    batch_size = 256

    with tf.name_scope('sigmoid_cross_entropy'):
        cross_entropy = tf.reduce_mean( tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=model), w_))

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)


    with tf.name_scope('accuracy'):
        # all labels
    
        correct_prediction = tf.boolean_mask(tf.equal( tf.round( y_ ), tf.round( model ) ), weighting)

        print correct_prediction
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # create a summary for our cost and accuracy
        tf.summary.scalar("cost", cross_entropy)
        tf.summary.scalar("accuracy", accuracy)

        # merge all summaries into a single "operation" which we can execute in a session 
        summary_op = tf.summary.merge_all()

        writer = tf.summary.FileWriter('./tmp/tensorflow_logs', graph=sess.graph)
        batch_count = train_data.shape[0] / batch_size

        for epoch in range(training_epochs):
            # number of batches in one epoch
            idxs = np.random.permutation(train_data.shape[0]) #shuffled ordering
            X_random = train_data[idxs]
            Y_random = train_label[idxs]
            train_phi_random = train_phi[idxs]
            train_M_random = train_M[idxs]


            for i in range(batch_count):
                train_data_batch = X_random[i * batch_size: (i+1) * batch_size,:]
                train_label_batch = Y_random[i * batch_size: (i+1) * batch_size,:]
                train_phi_batch = train_phi_random[i * batch_size: (i+1) * batch_size,:]
                train_M_batch = train_M_random[i * batch_size: (i+1) * batch_size,:]

                summary, _ = sess.run([summary_op, train_step], feed_dict={x: train_data_batch, y_: train_label_batch, w_: train_phi_batch, weighting: train_M_batch})
                # write log
                writer.add_summary(summary, epoch * batch_count + i)

                if i % 400 == 0:

                    train_accuracy, train_loss = sess.run((accuracy, cross_entropy),
                            feed_dict={x: train_data, y_: train_label, w_: train_phi, weighting: train_M})

                    train_accuracy_batches.append(train_accuracy)

                    #train_accuracy, train_loss = sess.run((accuracy, cross_entropy),
                    #    feed_dict={x: train_data_batch, y_: train_label_batch, w_: train_phi_batch, weighting: train_M_batch})

                    print('step %d, training accuracy %g, training loss: %g' %
                        (i, train_accuracy, train_loss))

        test_accuracy, test_loss = sess.run((accuracy, cross_entropy),
             feed_dict={x: test_data, y_: test_label, w_: test_phi, weighting: test_M})

        print('test accuracy %g, test loss: %g' %
            (test_accuracy, test_loss))          

        return batch_count, train_accuracy_batches


if __name__ == '__main__':

    cross_validation_dir = './cv_5_folds/'
    data_dir = './ExtraSensory.per_uuid_features_labels/'

    cv_files = [ ['fold_0_test_android_uuids.txt', 'fold_0_test_iphone_uuids.txt', 'fold_0_train_android_uuids.txt', 'fold_0_train_iphone_uuids.txt'], 
                ['fold_1_test_android_uuids.txt', 'fold_1_test_iphone_uuids.txt', 'fold_1_train_android_uuids.txt', 'fold_1_train_iphone_uuids.txt'],
                ['fold_2_test_android_uuids.txt', 'fold_2_test_iphone_uuids.txt', 'fold_2_train_android_uuids.txt', 'fold_2_train_iphone_uuids.txt'],
                ['fold_3_test_android_uuids.txt', 'fold_3_test_iphone_uuids.txt', 'fold_3_train_android_uuids.txt', 'fold_3_train_iphone_uuids.txt'],
                ['fold_4_test_android_uuids.txt', 'fold_4_test_iphone_uuids.txt', 'fold_4_train_android_uuids.txt', 'fold_4_train_iphone_uuids.txt']]

    sensors_to_use = ['Acc','Gyro', 'WAcc', 'Loc', 'Aud', 'PS'];
    target_labels = ['LYING_DOWN', 'SITTING', 'FIX_walking', 'FIX_running', 'BICYCLING', 'SLEEPING', 'LAB_WORK', 'IN_CLASS', 'IN_A_MEETING', 'LOC_main_workplace', 'OR_indoors', 'OR_outside', 'IN_A_CAR', 'ON_A_BUS', 'DRIVE_-_I_M_THE_DRIVER', 'DRIVE_-_I_M_A_PASSENGER', 'LOC_home', 'FIX_restaurant', 'PHONE_IN_POCKET', 'OR_exercise', 'COOKING', 'SHOPPING', 'STROLLING', 'DRINKING__ALCOHOL_', 'BATHING_-_SHOWER', 'CLEANING', 'DOING_LAUNDRY', 'WASHING_DISHES', 'WATCHING_TV', 'SURFING_THE_INTERNET', 'AT_A_PARTY', 'AT_A_BAR', 'LOC_beach', 'SINGING', 'TALKING', 'COMPUTER_WORK', 'EATING', 'TOILET', 'GROOMING', 'DRESSING', 'AT_THE_GYM', 'STAIRS_-_GOING_UP', 'STAIRS_-_GOING_DOWN', 'ELEVATOR', 'OR_standing', 'AT_SCHOOL', 'PHONE_IN_HAND', 'PHONE_IN_BAG', 'PHONE_ON_TABLE', 'WITH_CO-WORKERS', 'WITH_FRIENDS']
    feature_names = ['raw_acc:magnitude_stats:mean', 'raw_acc:magnitude_stats:std', 'raw_acc:magnitude_stats:moment3', 'raw_acc:magnitude_stats:moment4', 'raw_acc:magnitude_stats:percentile25', 'raw_acc:magnitude_stats:percentile50', 'raw_acc:magnitude_stats:percentile75', 'raw_acc:magnitude_stats:value_entropy', 'raw_acc:magnitude_stats:time_entropy', 'raw_acc:magnitude_spectrum:log_energy_band0', 'raw_acc:magnitude_spectrum:log_energy_band1', 'raw_acc:magnitude_spectrum:log_energy_band2', 'raw_acc:magnitude_spectrum:log_energy_band3', 'raw_acc:magnitude_spectrum:log_energy_band4', 'raw_acc:magnitude_spectrum:spectral_entropy', 'raw_acc:magnitude_autocorrelation:period', 'raw_acc:magnitude_autocorrelation:normalized_ac', 'raw_acc:3d:mean_x', 'raw_acc:3d:mean_y', 'raw_acc:3d:mean_z', 'raw_acc:3d:std_x', 'raw_acc:3d:std_y', 'raw_acc:3d:std_z', 'raw_acc:3d:ro_xy', 'raw_acc:3d:ro_xz', 'raw_acc:3d:ro_yz', 'proc_gyro:magnitude_stats:mean', 'proc_gyro:magnitude_stats:std', 'proc_gyro:magnitude_stats:moment3', 'proc_gyro:magnitude_stats:moment4', 'proc_gyro:magnitude_stats:percentile25', 'proc_gyro:magnitude_stats:percentile50', 'proc_gyro:magnitude_stats:percentile75', 'proc_gyro:magnitude_stats:value_entropy', 'proc_gyro:magnitude_stats:time_entropy', 'proc_gyro:magnitude_spectrum:log_energy_band0', 'proc_gyro:magnitude_spectrum:log_energy_band1', 'proc_gyro:magnitude_spectrum:log_energy_band2', 'proc_gyro:magnitude_spectrum:log_energy_band3', 'proc_gyro:magnitude_spectrum:log_energy_band4', 'proc_gyro:magnitude_spectrum:spectral_entropy', 'proc_gyro:magnitude_autocorrelation:period', 'proc_gyro:magnitude_autocorrelation:normalized_ac', 'proc_gyro:3d:mean_x', 'proc_gyro:3d:mean_y', 'proc_gyro:3d:mean_z', 'proc_gyro:3d:std_x', 'proc_gyro:3d:std_y', 'proc_gyro:3d:std_z', 'proc_gyro:3d:ro_xy', 'proc_gyro:3d:ro_xz', 'proc_gyro:3d:ro_yz', 'raw_magnet:magnitude_stats:mean', 'raw_magnet:magnitude_stats:std', 'raw_magnet:magnitude_stats:moment3', 'raw_magnet:magnitude_stats:moment4', 'raw_magnet:magnitude_stats:percentile25', 'raw_magnet:magnitude_stats:percentile50', 'raw_magnet:magnitude_stats:percentile75', 'raw_magnet:magnitude_stats:value_entropy', 'raw_magnet:magnitude_stats:time_entropy', 'raw_magnet:magnitude_spectrum:log_energy_band0', 'raw_magnet:magnitude_spectrum:log_energy_band1', 'raw_magnet:magnitude_spectrum:log_energy_band2', 'raw_magnet:magnitude_spectrum:log_energy_band3', 'raw_magnet:magnitude_spectrum:log_energy_band4', 'raw_magnet:magnitude_spectrum:spectral_entropy', 'raw_magnet:magnitude_autocorrelation:period', 'raw_magnet:magnitude_autocorrelation:normalized_ac', 'raw_magnet:3d:mean_x', 'raw_magnet:3d:mean_y', 'raw_magnet:3d:mean_z', 'raw_magnet:3d:std_x', 'raw_magnet:3d:std_y', 'raw_magnet:3d:std_z', 'raw_magnet:3d:ro_xy', 'raw_magnet:3d:ro_xz', 'raw_magnet:3d:ro_yz', 'raw_magnet:avr_cosine_similarity_lag_range0', 'raw_magnet:avr_cosine_similarity_lag_range1', 'raw_magnet:avr_cosine_similarity_lag_range2', 'raw_magnet:avr_cosine_similarity_lag_range3', 'raw_magnet:avr_cosine_similarity_lag_range4', 'watch_acceleration:magnitude_stats:mean', 'watch_acceleration:magnitude_stats:std', 'watch_acceleration:magnitude_stats:moment3', 'watch_acceleration:magnitude_stats:moment4', 'watch_acceleration:magnitude_stats:percentile25', 'watch_acceleration:magnitude_stats:percentile50', 'watch_acceleration:magnitude_stats:percentile75', 'watch_acceleration:magnitude_stats:value_entropy', 'watch_acceleration:magnitude_stats:time_entropy', 'watch_acceleration:magnitude_spectrum:log_energy_band0', 'watch_acceleration:magnitude_spectrum:log_energy_band1', 'watch_acceleration:magnitude_spectrum:log_energy_band2', 'watch_acceleration:magnitude_spectrum:log_energy_band3', 'watch_acceleration:magnitude_spectrum:log_energy_band4', 'watch_acceleration:magnitude_spectrum:spectral_entropy', 'watch_acceleration:magnitude_autocorrelation:period', 'watch_acceleration:magnitude_autocorrelation:normalized_ac', 'watch_acceleration:3d:mean_x', 'watch_acceleration:3d:mean_y', 'watch_acceleration:3d:mean_z', 'watch_acceleration:3d:std_x', 'watch_acceleration:3d:std_y', 'watch_acceleration:3d:std_z', 'watch_acceleration:3d:ro_xy', 'watch_acceleration:3d:ro_xz', 'watch_acceleration:3d:ro_yz', 'watch_acceleration:spectrum:x_log_energy_band0', 'watch_acceleration:spectrum:x_log_energy_band1', 'watch_acceleration:spectrum:x_log_energy_band2', 'watch_acceleration:spectrum:x_log_energy_band3', 'watch_acceleration:spectrum:x_log_energy_band4', 'watch_acceleration:spectrum:y_log_energy_band0', 'watch_acceleration:spectrum:y_log_energy_band1', 'watch_acceleration:spectrum:y_log_energy_band2', 'watch_acceleration:spectrum:y_log_energy_band3', 'watch_acceleration:spectrum:y_log_energy_band4', 'watch_acceleration:spectrum:z_log_energy_band0', 'watch_acceleration:spectrum:z_log_energy_band1', 'watch_acceleration:spectrum:z_log_energy_band2', 'watch_acceleration:spectrum:z_log_energy_band3', 'watch_acceleration:spectrum:z_log_energy_band4', 'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range0', 'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range1', 'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range2', 'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range3', 'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range4', 'watch_heading:mean_cos', 'watch_heading:std_cos', 'watch_heading:mom3_cos', 'watch_heading:mom4_cos', 'watch_heading:mean_sin', 'watch_heading:std_sin', 'watch_heading:mom3_sin', 'watch_heading:mom4_sin', 'watch_heading:entropy_8bins', 'location:num_valid_updates', 'location:log_latitude_range', 'location:log_longitude_range', 'location:min_altitude', 'location:max_altitude', 'location:min_speed', 'location:max_speed', 'location:best_horizontal_accuracy', 'location:best_vertical_accuracy', 'location:diameter', 'location:log_diameter', 'location_quick_features:std_lat', 'location_quick_features:std_long', 'location_quick_features:lat_change', 'location_quick_features:long_change', 'location_quick_features:mean_abs_lat_deriv', 'location_quick_features:mean_abs_long_deriv', 'audio_naive:mfcc0:mean', 'audio_naive:mfcc1:mean', 'audio_naive:mfcc2:mean', 'audio_naive:mfcc3:mean', 'audio_naive:mfcc4:mean', 'audio_naive:mfcc5:mean', 'audio_naive:mfcc6:mean', 'audio_naive:mfcc7:mean', 'audio_naive:mfcc8:mean', 'audio_naive:mfcc9:mean', 'audio_naive:mfcc10:mean', 'audio_naive:mfcc11:mean', 'audio_naive:mfcc12:mean', 'audio_naive:mfcc0:std', 'audio_naive:mfcc1:std', 'audio_naive:mfcc2:std', 'audio_naive:mfcc3:std', 'audio_naive:mfcc4:std', 'audio_naive:mfcc5:std', 'audio_naive:mfcc6:std', 'audio_naive:mfcc7:std', 'audio_naive:mfcc8:std', 'audio_naive:mfcc9:std', 'audio_naive:mfcc10:std', 'audio_naive:mfcc11:std', 'audio_naive:mfcc12:std', 'audio_properties:max_abs_value', 'audio_properties:normalization_multiplier', 'discrete:app_state:is_active', 'discrete:app_state:is_inactive', 'discrete:app_state:is_background', 'discrete:app_state:missing', 'discrete:battery_plugged:is_ac', 'discrete:battery_plugged:is_usb', 'discrete:battery_plugged:is_wireless', 'discrete:battery_plugged:missing', 'discrete:battery_state:is_unknown', 'discrete:battery_state:is_unplugged', 'discrete:battery_state:is_not_charging', 'discrete:battery_state:is_discharging', 'discrete:battery_state:is_charging', 'discrete:battery_state:is_full', 'discrete:battery_state:missing', 'discrete:on_the_phone:is_False', 'discrete:on_the_phone:is_True', 'discrete:on_the_phone:missing', 'discrete:ringer_mode:is_normal', 'discrete:ringer_mode:is_silent_no_vibrate', 'discrete:ringer_mode:is_silent_with_vibrate', 'discrete:ringer_mode:missing', 'discrete:wifi_status:is_not_reachable', 'discrete:wifi_status:is_reachable_via_wifi', 'discrete:wifi_status:is_reachable_via_wwan', 'discrete:wifi_status:missing', 'lf_measurements:light', 'lf_measurements:pressure', 'lf_measurements:proximity_cm', 'lf_measurements:proximity', 'lf_measurements:relative_humidity', 'lf_measurements:battery_level', 'lf_measurements:screen_brightness', 'lf_measurements:temperature_ambient', 'discrete:time_of_day:between0and6', 'discrete:time_of_day:between3and9', 'discrete:time_of_day:between6and12', 'discrete:time_of_day:between9and15', 'discrete:time_of_day:between12and18', 'discrete:time_of_day:between15and21', 'discrete:time_of_day:between18and24', 'discrete:time_of_day:between21and3']


    feat_sensor_names = get_sensor_names_from_features(feature_names);

    cross_validation_epoch = 1

    for i in range(cross_validation_epoch):
        print i
        train_data = []
        train_label = []
        train_M = []

        test_data = []
        test_label = []
        test_M = []
        
        # train data 
        for f in cv_files[0:i] + cv_files[i+1:]:

            for cv in f:
                fold =  cross_validation_dir +  cv

                for uuid in open(fold, 'r'):
                    uuid = data_dir + uuid[:-1]
                    (X,Y,M,timestamps,feature_names,label_names) = read_user_data(uuid);

                    X_train = project_features_to_selected_sensors(X, feat_sensor_names,sensors_to_use)
                    (mean_vec,std_vec) = estimate_standardization_params(X_train);
                    X_train = standardize_features(X_train,mean_vec,std_vec);
                    
                    X_train[np.isnan(X_train)] = 0.

                    train_data.append(X_train)
                    train_label.append(Y)
                    train_M.append(M)
        
        train_data = np.concatenate(train_data)
        train_label = np.concatenate(train_label)
        train_M = np.concatenate(train_M)
        train_phi = np.zeros(train_label.shape)
        
        one_idx = train_M == True
        zero_idx = train_M == False

        pos_num = sum(train_label == 1)
        neg_num = sum(train_label == 0)

        pos_frac = pos_num / (train_label.shape[0]+0.0)
        neg_frac = neg_num / (train_label.shape[0]+0.0)


        for col in range(train_label.shape[1]):
            for row in range(train_label.shape[0]):
                if train_label[row, col] == 1:
                    train_phi[row, col] = 1.0 / pos_frac[col]
                else:
                    train_phi[row, col] = 1.0 / neg_frac[col]

        train_phi[one_idx] = 0.0
        
    
        # test data
        for f in cv_files[i]:
            fold =  cross_validation_dir +  f
            for uuid in open(fold, 'r'):

                uuid = data_dir + uuid[:-1]
                (X,Y,M,timestamps,feature_names,label_names) = read_user_data(uuid);

                X_test = project_features_to_selected_sensors(X, feat_sensor_names,sensors_to_use)
                (mean_vec,std_vec) = estimate_standardization_params(X_test);
                X_test = standardize_features(X_test,mean_vec,std_vec);
                
                X_test[np.isnan(X_test)] = 0.

                test_data.append(X_test)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                test_label.append(Y)
                test_M.append(M)


        test_data = np.array(test_data[0])
        test_label = np.array(test_label[0])
        test_M = np.array(test_M[0])
        test_phi = np.zeros(test_M.shape)

        print test_data.shape

        one_idx = test_label == 1
        pos_num = sum(test_label == 1)
        neg_num = sum(test_label == 0)

        pos_frac = pos_num / (test_label.shape[0]+0.0)
        neg_frac = neg_num / (test_label.shape[0]+0.0)


        for col in range(test_label.shape[1]):
            for row in range(test_label.shape[0]):
                if test_label[row, col] == True:
                    test_phi[row, col] = 1.0 / pos_frac[col]
                else:
                    test_phi[row, col] = 1.0 / neg_frac[col]

        test_phi[one_idx] = 0.0

        
        train_M = np.logical_not(train_M)
        test_M = np.logical_not(test_M)
        print train_M.shape
        print test_M.shape

        batch_count, train_accuracy_batches = main(train_data, train_phi, train_label, train_M, test_data, test_phi, test_label, test_M)                                                        
		

'''
x = range(batch_count)
x_ = range(0, batch_count, 400)

plt.plot(x_, train_accuracy_batches,'ro-')
plt.plot(x_, test_accuracy_batches, 'bo-')

plt.xticks(np.arange(min(x), (max(x)+1), 400.0))
plt.legend(['train accuracy', 'test accuracy'], loc='lower right')
plt.xlabel('Batches', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.title('Fully connected Network', fontsize=15)
plt.show()

'''
