import os
import re

import pandas as pd
from pfs.utils.opdb import opDB

pd.set_option('display.max_colwidth', -1)


def stripField(rawCmd, field):
    """ Strip given text field from rawCmd """
    if re.search(field, rawCmd) is None:
        return rawCmd
    idlm = re.search(field, rawCmd).span(0)[-1]
    sub = rawCmd[idlm:]
    sub = sub if sub.find(' ') == -1 else sub[:sub.find(' ')]
    pattern = f' {field}{sub[0]}(.*?){sub[0]}' if sub[0] in ['"', "'"] else f' {field}{sub}'
    m = re.search(pattern, rawCmd)
    return rawCmd.replace(m.group(), '').strip()


sql_all = "select iic_sequence.visit_set_id, sequence_type, name, comments, sps_exposure.pfs_visit_id, cmd_str, exp_type, sps_module_id, arm, notes, data_flag,time_exp_start,status_flag,cmd_output from sps_exposure \
inner join visit_set on sps_exposure.pfs_visit_id=visit_set.pfs_visit_id \
inner join iic_sequence on visit_set.visit_set_id=iic_sequence.visit_set_id \
inner join sps_visit on sps_exposure.pfs_visit_id=sps_visit.pfs_visit_id \
inner join sps_camera on sps_exposure.sps_camera_id = sps_camera.sps_camera_id \
left outer join iic_sequence_status on iic_sequence.visit_set_id=iic_sequence_status.visit_set_id \
left outer join sps_annotation on sps_exposure.pfs_visit_id=sps_annotation.pfs_visit_id order by sps_exposure.pfs_visit_id desc"

allColumns = ['visit_set_id', 'sequence_type', 'name', 'comments', 'pfs_visit_id', 'cmd_str', 'exp_type',
              'sps_module_id', 'arm', 'notes', 'data_flag', 'time_exp_start', 'status', 'output']

opdbColumns = ['visit_set_id', 'sequence_type', 'name', 'comments', 'visitStart', 'visitEnd', 'cmd_str', 'startdate',
               'status', 'output']


def buildSequenceSummary(allRows):
    logs = []
    for (visit_set_id, sequence_type, name, comments, cmd_str, status, output), visit_set in allRows.groupby(
            ['visit_set_id', 'sequence_type', 'name', 'comments', 'cmd_str', 'status', 'output']):
        status = 'OK' if status == 0 else 'FAILED'
        logs.append((visit_set_id, sequence_type, name, comments, visit_set.pfs_visit_id.min(),
                     visit_set.pfs_visit_id.max(), stripField(stripField(cmd_str, 'comments='), 'name='),
                     visit_set.time_exp_start.min(), status, output))

    return pd.DataFrame(logs, columns=opdbColumns).set_index('visit_set_id').sort_index(ascending=False)


def spsLogbook(directory):
    allRows = pd.DataFrame(opDB.fetchall(sql_all), columns=allColumns)
    opdbLogs = buildSequenceSummary(allRows)
    output = os.path.join(directory, 'index.html')
    opdbLogs.to_html(output)
    return output


def main():
    """Command-line interface to create PfsDesign files"""
    import argparse

    parser = argparse.ArgumentParser(description="Create an html log page for SPS opdb ")
    parser.add_argument("--directory", default=".", help="Directory in which to write html file")

    args = parser.parse_args()
    output = spsLogbook(args.directory)

    print("Wrote %s" % output)


if __name__ == "__main__":
    main()
