import ProjectEulerSolutions.Lib.CubicFiveClass.Block000
import ProjectEulerSolutions.Lib.CubicFiveClass.Block001
import ProjectEulerSolutions.Lib.CubicFiveClass.Block002
import ProjectEulerSolutions.Lib.CubicFiveClass.Block003
import ProjectEulerSolutions.Lib.CubicFiveClass.Block004
import ProjectEulerSolutions.Lib.CubicFiveClass.Block005
import ProjectEulerSolutions.Lib.CubicFiveClass.Block006
import ProjectEulerSolutions.Lib.CubicFiveClass.Block007
import ProjectEulerSolutions.Lib.CubicFiveClass.Block008
import ProjectEulerSolutions.Lib.CubicFiveClass.Block009

/-! One complete digit class proves existence without executing either solver. -/
namespace ProjectEulerSolutions.CubicFiveClass
open CubicClassWitness CubicSearch CubicGroupCounts CubicWindows
set_option Elab.async false

theorem exact_roots : rootsWith digits 1 9999 = [5027, 7061, 7202, 8288, 8384] := by
  have joined0 := rootsWith_join digits 1 1024 1024
    [] [] Block000.checked Block001.checked
  have joined1 := rootsWith_join digits 3073 1024 1024
    [] [5027] Block003.checked Block004.checked
  have joined2 := rootsWith_join digits 2049 1024 2048
    [] [5027] Block002.checked joined1
  have joined3 := rootsWith_join digits 1 2048 3072
    [] [5027] joined0 joined2
  have joined4 := rootsWith_join digits 5121 1024 1024
    [] [7061] Block005.checked Block006.checked
  have joined5 := rootsWith_join digits 8193 1024 783
    [8288, 8384] [] Block008.checked Block009.checked
  have joined6 := rootsWith_join digits 7169 1024 1807
    [7202] [8288, 8384] Block007.checked joined5
  have joined7 := rootsWith_join digits 5121 2048 2831
    [7061] [7202, 8288, 8384] joined4 joined6
  have joined8 := rootsWith_join digits 1 5120 4879
    [5027] [7061, 7202, 8288, 8384] joined3 joined7
  exact joined8

theorem five_count :
    ((rootsWith (CubicPermutations.signature (cube 5027)) 1 9999).length : Int) = 5 := by
  rw [example_signature, exact_roots]
  rfl

theorem model_returns : Returns 5 start (choose 5 (collect (prior 10000))) :=
  returns_of_count 10000 5027 5 (by decide) (by decide) (by decide) five_count

end ProjectEulerSolutions.CubicFiveClass
