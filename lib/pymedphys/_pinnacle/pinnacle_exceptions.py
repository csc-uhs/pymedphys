# Copyright (C) 2019 South Western Sydney Local Health District,
# University of New South Wales

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This work is derived from:
# https://github.com/AndrewWAlexander/Pinnacle-tar-DICOM
# which is released under the following license:

# Copyright (c) [2017] [Colleen Henschel, Andrew Alexander]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class MissingCTImageError(Exception):
    # Raised when a plan does not have an associated primary CT image
    pass


class MissingTrialBeamsError(Exception):
    # Raised when a trial does not have any beams associated with it
    pass


class MissingBeamDoseError(Exception):
    # Raised when all plan beams are missing dose
    pass


class InvalidDoseNormalizationError(Exception):
    # Raised when a beam has a non-zero prescription dose but the
    # interpolated dose at its prescription point is zero, so its
    # monitor units cannot be derived. Continuing would silently
    # under-report the PLAN dose summation, so the trial's
    # dose export is failed instead.
    pass


class IsocenterNotFoundError(Exception):
    # Raised when no isocenter can be resolved for a beam — either the
    # trial names an isocenter point that does not exist in plan.Points,
    # or no isocenter-like point can be identified at all.  Assuming an
    # arbitrary point is dangerous, so the trial's RTPLAN export is
    # failed instead.
    pass


class MachineDataNotFoundError(Exception):
    # Raised when required machine geometry (currently the MLC leaf
    # boundary layout) cannot be derived from plan.Pinnacle.Machines for
    # a beam that uses an MLC.  Exporting with an assumed or default
    # (e.g. Varian-Millennium) boundary table is dangerous — a wrong table
    # silently shifts every leaf pair — so the trial's RTPLAN export is
    # failed instead.
    pass


class UnsupportedWedgeError(Exception):
    # Raised when a beam uses a wedge this exporter cannot convert
    # correctly.  The motorized wedge is supported, having been validated
    # against a Pinnacle RTPLAN export.  Other wedge types, lateral wedge
    # orientations, an undeterminable wedge angle, and machines whose
    # output factor table has no entry for the wedge in use are all
    # refused: each would produce a plan that looks complete while
    # misdescribing the wedge or its monitor units, which no inspection
    # of the converted plan would reveal.
    pass
