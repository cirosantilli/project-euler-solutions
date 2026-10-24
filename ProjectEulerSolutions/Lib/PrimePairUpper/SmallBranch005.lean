import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates1007 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1007_checked : candidates edge 5 0 8017 tail1008 best.bound = rootCandidates1007 := by
  unfold candidates
  have row : edge 8017 = SearchData.row8017.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1006 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1006_checked : candidates edge 5 0 8011 tail1007 best.bound = rootCandidates1006 := by
  unfold candidates
  have row : edge 8011 = SearchData.row8011.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1005 : List Nat := [8093, 8147]
set_option maxRecDepth 4096 in
theorem rootCandidates1005_checked : candidates edge 5 0 8009 tail1006 best.bound = rootCandidates1005 := by
  unfold candidates
  have row : edge 8009 = SearchData.row8009.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1004 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1004_checked : candidates edge 5 0 7993 tail1005 best.bound = rootCandidates1004 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1003 : List Nat := [8011]
set_option maxRecDepth 4096 in
theorem rootCandidates1003_checked : candidates edge 5 0 7963 tail1004 best.bound = rootCandidates1003 := by
  unfold candidates
  have row : edge 7963 = SearchData.row7963.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1002 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1002_checked : candidates edge 5 0 7951 tail1003 best.bound = rootCandidates1002 := by
  unfold candidates
  have row : edge 7951 = SearchData.row7951.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1001 : List Nat := [8219, 8297, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates1001_checked : candidates edge 5 0 7949 tail1002 best.bound = rootCandidates1001 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1000 : List Nat := [8081]
set_option maxRecDepth 4096 in
theorem rootCandidates1000_checked : candidates edge 5 0 7937 tail1001 best.bound = rootCandidates1000 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
