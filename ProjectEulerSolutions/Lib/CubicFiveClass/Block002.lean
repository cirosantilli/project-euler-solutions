import ProjectEulerSolutions.Lib.CubicFiveClass.Digits

/-! Independent small checks bound kernel memory for class completeness. -/
namespace ProjectEulerSolutions.CubicFiveClass.Block002
open CubicClassWitness
set_option Elab.async false
-- Kernel reduction checks at most 128 root signatures per declaration.
set_option maxRecDepth 4096

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part0 : rootsWith digits 2049 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part1 : rootsWith digits 2177 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part2 : rootsWith digits 2305 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part3 : rootsWith digits 2433 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part4 : rootsWith digits 2561 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part5 : rootsWith digits 2689 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part6 : rootsWith digits 2817 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part7 : rootsWith digits 2945 128 = [] := by decide

theorem checked : rootsWith digits 2049 1024 = [] := by
  have joined0 := rootsWith_join digits 2049 128 128
    [] [] part0 part1
  have joined1 := rootsWith_join digits 2305 128 128
    [] [] part2 part3
  have joined2 := rootsWith_join digits 2049 256 256
    [] [] joined0 joined1
  have joined3 := rootsWith_join digits 2561 128 128
    [] [] part4 part5
  have joined4 := rootsWith_join digits 2817 128 128
    [] [] part6 part7
  have joined5 := rootsWith_join digits 2561 256 256
    [] [] joined3 joined4
  have joined6 := rootsWith_join digits 2049 512 512
    [] [] joined2 joined5
  exact joined6

end ProjectEulerSolutions.CubicFiveClass.Block002
