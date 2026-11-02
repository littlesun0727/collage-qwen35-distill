import unittest
from unittest.mock import patch
from urllib.error import URLError
import local_transport as client

class TransportTests(unittest.TestCase):
    def test_model_mismatch_never_starts_service(self):
        with patch.object(client,'control',return_value={'model':'Qwen3.5-9B','status':'ready'}) as ctl:
            with self.assertRaises(ValueError): client.ready('Qwen3.5-4B')
            self.assertEqual(ctl.call_count,1)

    def test_killed_service_ensured_during_request(self):
        with patch.object(client,'control',side_effect=[{'model':'Qwen3.5-4B','status':'stopped'},
                    {'model':'Qwen3.5-4B','status':'ready','port':8000}]) as ctl:
            self.assertIn(':8000/',client.ready('Qwen3.5-4B'))
            self.assertEqual([c.args[0] for c in ctl.call_args_list],['status','ensure'])

    def test_failed_http_checks_service_again_bounded(self):
        with patch.object(client,'ready',return_value='http://127.0.0.1:8000/v1/chat/completions') as ready, \
             patch.object(client,'urlopen',side_effect=URLError('killed')),patch.object(client.time,'sleep'):
            with self.assertRaises(RuntimeError): client.infer({'model':'Qwen3.5-4B'})
            self.assertEqual(ready.call_count,3)
