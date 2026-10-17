import subprocess
import unittest
from unittest.mock import patch

from backend.app.hardware.exposure.controller import ExposureController


class ExposureControllerTests(unittest.TestCase):
    def run_controller(self, options, modern=False, reject_temperature=False):
        auto = 'auto_exposure' if modern else 'exposure_auto'
        shutter = 'exposure_time_absolute' if modern else 'exposure_absolute'
        wb = 'white_balance_automatic' if modern else 'white_balance_temperature_auto'
        values = {auto: 3, shutter: 5, wb: 1, 'white_balance_temperature': 3300,
                  'power_line_frequency': 2, 'gain': 0}
        writes = []

        def run(command, **kwargs):
            flag = command[3]
            if flag == '--list-ctrls':
                result = '\n'.join(f'{k} 0x00900000 (int) : min=5 max=2500 value={v}' for k, v in values.items())
            elif flag == '-c':
                key, value = command[4].split('=')
                if reject_temperature and key == 'white_balance_temperature':
                    raise subprocess.CalledProcessError(1, command, stderr='Device rejected temperature')
                values[key] = int(value)
                writes.append((key, int(value)))
                result = ''
            else:
                key = command[4]
                result = f'{key}: {values[key]}'
            return subprocess.CompletedProcess(command, 0, stdout=result, stderr='')

        with patch('backend.app.hardware.exposure.controller.subprocess.run', side_effect=run):
            result = ExposureController()._apply_linux(0, 50, 100, False, options)
        return result, writes, wb

    def test_manual_temperature_disables_awb_first_on_both_driver_names(self):
        for modern in (False, True):
            with self.subTest(modern=modern):
                result, writes, wb = self.run_controller({'white_balance_temperature': 4500}, modern)
                self.assertLess(writes.index((wb, 0)), writes.index(('white_balance_temperature', 4500)))
                self.assertEqual(result['readback']['white_balance_temperature'], 4500)
                self.assertFalse(result['errors'])

    def test_explicit_auto_does_not_write_inactive_temperature(self):
        result, writes, wb = self.run_controller({'white_balance_automatic': 1, 'white_balance_temperature': 4500})
        self.assertNotIn(('white_balance_temperature', 4500), writes)
        self.assertFalse(result['controls']['white_balance_temperature'])
        self.assertIn('requires', result['errors']['white_balance_temperature'])

    def test_unconfigured_white_balance_is_observed_and_preserved(self):
        result, writes, wb = self.run_controller({})
        self.assertFalse(any(key == wb for key, _ in writes))
        self.assertEqual(result['observed_controls'][wb], 1)
        self.assertEqual(result['observed_controls']['white_balance_temperature'], 3300)
        self.assertEqual(result['readback']['exposure_absolute'], 100)

    def test_rejected_temperature_is_reported(self):
        result, _, _ = self.run_controller({'white_balance_temperature': 4500}, reject_temperature=True)
        self.assertFalse(result['controls']['white_balance_temperature'])
        self.assertIn('rejected', result['errors']['white_balance_temperature'])


if __name__ == '__main__':
    unittest.main()
