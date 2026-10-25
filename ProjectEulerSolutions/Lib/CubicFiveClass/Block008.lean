import ProjectEulerSolutions.Lib.CubicFiveClass.Digits

/-! Independent small checks bound kernel memory for class completeness. -/
namespace ProjectEulerSolutions.CubicFiveClass.Block008
open CubicClassWitness
set_option Elab.async false
-- Kernel reduction checks at most 128 root signatures per declaration.
set_option maxRecDepth 4096

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part0 : rootsWith digits 8193 128 = [8288] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part1 : rootsWith digits 8321 128 = [8384] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part2 : rootsWith digits 8449 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part3 : rootsWith digits 8577 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part4 : rootsWith digits 8705 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part5 : rootsWith digits 8833 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part6 : rootsWith digits 8961 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part7 : rootsWith digits 9089 128 = [] := by decide

theorem checked : rootsWith digits 8193 1024 = [8288, 8384] := by
  have joined0 := rootsWith_join digits 8193 128 128
    [8288] [8384] part0 part1
  have joined1 := rootsWith_join digits 8449 128 128
    [] [] part2 part3
  have joined2 := rootsWith_join digits 8193 256 256
    [8288, 8384] [] joined0 joined1
  have joined3 := rootsWith_join digits 8705 128 128
    [] [] part4 part5
  have joined4 := rootsWith_join digits 8961 128 128
    [] [] part6 part7
  have joined5 := rootsWith_join digits 8705 256 256
    [] [] joined3 joined4
  have joined6 := rootsWith_join digits 8193 512 512
    [8288, 8384] [] joined2 joined5
  exact joined6

end ProjectEulerSolutions.CubicFiveClass.Block008
