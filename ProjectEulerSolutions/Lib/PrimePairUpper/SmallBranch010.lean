import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates967 : List Nat := [7883, 8363]
set_option maxRecDepth 4096 in
theorem rootCandidates967_checked : candidates edge 5 0 7643 tail968 best.bound = rootCandidates967 := by
  unfold candidates
  have row : edge 7643 = SearchData.row7643.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates966 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates966_checked : candidates edge 5 0 7639 tail967 best.bound = rootCandidates966 := by
  unfold candidates
  have row : edge 7639 = SearchData.row7639.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates965 : List Nat := [7741, 8269]
set_option maxRecDepth 4096 in
theorem rootCandidates965_checked : candidates edge 5 0 7621 tail966 best.bound = rootCandidates965 := by
  unfold candidates
  have row : edge 7621 = SearchData.row7621.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates964 : List Nat := [7649, 7853]
set_option maxRecDepth 4096 in
theorem rootCandidates964_checked : candidates edge 5 0 7607 tail965 best.bound = rootCandidates964 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates963 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates963_checked : candidates edge 5 0 7603 tail964 best.bound = rootCandidates963 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates962 : List Nat := [7867]
set_option maxRecDepth 4096 in
theorem rootCandidates962_checked : candidates edge 5 0 7591 tail963 best.bound = rootCandidates962 := by
  unfold candidates
  have row : edge 7591 = SearchData.row7591.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates961 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem rootCandidates961_checked : candidates edge 5 0 7589 tail962 best.bound = rootCandidates961 := by
  unfold candidates
  have row : edge 7589 = SearchData.row7589.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates960 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem rootCandidates960_checked : candidates edge 5 0 7583 tail961 best.bound = rootCandidates960 := by
  unfold candidates
  have row : edge 7583 = SearchData.row7583.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
