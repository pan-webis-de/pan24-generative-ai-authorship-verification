# Generated by the protocol buffer compiler.  DO NOT EDIT!
# sources: gnews_url.proto
# plugin: python-betterproto
from dataclasses import dataclass

import betterproto


@dataclass
class GNewsURL(betterproto.Message):
    tag: int = betterproto.int64_field(1)
    url: str = betterproto.string_field(4)
    amp_url: str = betterproto.string_field(26)
