import os
import helper_functions as tlbx
import AttenVis_config as cfg

participants_df, participants_to_study = tlbx.load_participants()
check_participants = []
for participant in participants_to_study:
    diagnosis, study,visit_dir,subjID_date = tlbx.read_participant_details_from_dataframe(participants_df,participant)
    participant_recon_dir = cfg.subj_dir +  subjID_date
    visit_date = os.path.basename(visit_dir).split('_')[1]
    transfile = tlbx.find_files( '_'.join([participant,visit_date,'trans.fif']),participant_recon_dir)
    if not transfile:
        print(f"Transfile not found for participant {participant}, visit {visit_date}")
        check_participants.append(participant)
