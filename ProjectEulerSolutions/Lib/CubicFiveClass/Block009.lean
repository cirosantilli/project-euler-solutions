import ProjectEulerSolutions.Lib.CubicFiveClass.Digits

/-! Independent small checks bound kernel memory for class completeness. -/
namespace ProjectEulerSolutions.CubicFiveClass.Block009
open CubicClassWitness
set_option Elab.async false
-- Kernel reduction checks at most 128 root signatures per declaration.
set_option maxRecDepth 4096

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part0 : rootsWith digits 9217 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part1 : rootsWith digits 9345 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part2 : rootsWith digits 9473 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part3 : rootsWith digits 9601 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part4 : rootsWith digits 9729 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part5 : rootsWith digits 9857 128 = [] := by decide

set_option maxHeartbeats 1000000 in
-- Each check reduces at most 128 cube signatures and their digit comparisons.
theorem part6 : rootsWith digits 9985 15 = [] := by decide

theorem checked : rootsWith digits 9217 783 = [] := by
  have joined0 := rootsWith_join digits 9345 128 128
    [] [] part1 part2
  have joined1 := rootsWith_join digits 9217 128 256
    [] [] part0 joined0
  have joined2 := rootsWith_join digits 9601 128 128
    [] [] part3 part4
  have joined3 := rootsWith_join digits 9857 128 15
    [] [] part5 part6
  have joined4 := rootsWith_join digits 9601 256 143
    [] [] joined2 joined3
  have joined5 := rootsWith_join digits 9217 384 399
    [] [] joined1 joined4
  exact joined5

end ProjectEulerSolutions.CubicFiveClass.Block009
