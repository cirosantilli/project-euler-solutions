import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part000
import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part001
import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part002
import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part003
import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part004
import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part005
import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part006
import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part007
import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part008
import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part009
import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part010
import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part011
import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part012
import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part013
import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part014
import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part015
import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part016
import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part017
import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part018
import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part019
import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part020
import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part021
import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part022
import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part023
import ProjectEulerSolutions.Lib.PolygonalCycleOptimized

namespace ProjectEulerSolutions.PolygonalCycle.Reference
theorem results_checked : starts.map visit = results := by
  change (Part000.starts ++ Part001.starts ++ Part002.starts ++ Part003.starts ++ Part004.starts ++ Part005.starts ++ Part006.starts ++ Part007.starts ++ Part008.starts ++ Part009.starts ++ Part010.starts ++ Part011.starts ++ Part012.starts ++ Part013.starts ++ Part014.starts ++ Part015.starts ++ Part016.starts ++ Part017.starts ++ Part018.starts ++ Part019.starts ++ Part020.starts ++ Part021.starts ++ Part022.starts ++ Part023.starts).map visit = Part000.results ++ Part001.results ++ Part002.results ++ Part003.results ++ Part004.results ++ Part005.results ++ Part006.results ++ Part007.results ++ Part008.results ++ Part009.results ++ Part010.results ++ Part011.results ++ Part012.results ++ Part013.results ++ Part014.results ++ Part015.results ++ Part016.results ++ Part017.results ++ Part018.results ++ Part019.results ++ Part020.results ++ Part021.results ++ Part022.results ++ Part023.results
  simp only [List.map_append, Part000.checked, Part001.checked, Part002.checked, Part003.checked, Part004.checked, Part005.checked, Part006.checked, Part007.checked, Part008.checked, Part009.checked, Part010.checked, Part011.checked, Part012.checked, Part013.checked, Part014.checked, Part015.checked, Part016.checked, Part017.checked, Part018.checked, Part019.checked, Part020.checked, Part021.checked, Part022.checked, Part023.checked]

private theorem fold_results (values : List Int) (bound : Int) :
    values.foldl (fun best start => max best (visit start)) bound =
      (values.map visit).foldl max bound := by
  induction values generalizing bound with
  | nil => rfl
  | cons value rest ih => exact ih (max bound (visit value))

set_option maxRecDepth 4096 in
theorem reference_checked : reference 140 = 28684 := by
  unfold reference
  rw [← starts_checked]
  change starts.foldl (fun best start => max best (visit start)) 0 = 28684
  rw [fold_results, results_checked]
  decide

theorem models_equal : answer = reference 140 := by
  rw [answer_checked, reference_checked]
end ProjectEulerSolutions.PolygonalCycle.Reference
