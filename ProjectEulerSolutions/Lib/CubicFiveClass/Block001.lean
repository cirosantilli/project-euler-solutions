import ProjectEulerSolutions.Lib.CubicFiveClass.Digits

/-! Independent small checks bound kernel memory for class completeness. -/
namespace ProjectEulerSolutions.CubicFiveClass.Block001
open CubicClassWitness
set_option Elab.async false
-- Kernel reduction checks at most 128 root signatures per declaration.
set_option maxRecDepth 4096

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part0 : rootsWith digits 1025 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part1 : rootsWith digits 1153 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part2 : rootsWith digits 1281 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part3 : rootsWith digits 1409 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part4 : rootsWith digits 1537 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part5 : rootsWith digits 1665 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part6 : rootsWith digits 1793 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part7 : rootsWith digits 1921 128 = [] := by decide

theorem checked : rootsWith digits 1025 1024 = [] := by
  have joined0 := rootsWith_join digits 1025 128 128
    [] [] part0 part1
  have joined1 := rootsWith_join digits 1281 128 128
    [] [] part2 part3
  have joined2 := rootsWith_join digits 1025 256 256
    [] [] joined0 joined1
  have joined3 := rootsWith_join digits 1537 128 128
    [] [] part4 part5
  have joined4 := rootsWith_join digits 1793 128 128
    [] [] part6 part7
  have joined5 := rootsWith_join digits 1537 256 256
    [] [] joined3 joined4
  have joined6 := rootsWith_join digits 1025 512 512
    [] [] joined2 joined5
  exact joined6

end ProjectEulerSolutions.CubicFiveClass.Block001
