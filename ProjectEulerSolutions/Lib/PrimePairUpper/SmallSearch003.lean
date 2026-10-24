import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch002
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch003

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan1023 : scan edge (search edge 4) 5 [] 0 tail1023 best = best := by
  have cons : tail1023 = 8171 :: tail1024 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1023_checked]
  change scan edge (search edge 4) 5 [] 0 tail1024
    (if 4 ≤ rootCandidates1023.length then search edge 4 [8171] rootCandidates1023 8171 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1024

set_option maxRecDepth 4096 in
theorem outerScan1022 : scan edge (search edge 4) 5 [] 0 tail1022 best = best := by
  have cons : tail1022 = 8167 :: tail1023 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1022_checked]
  change scan edge (search edge 4) 5 [] 0 tail1023
    (if 4 ≤ rootCandidates1022.length then search edge 4 [8167] rootCandidates1022 8167 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1023

set_option maxRecDepth 4096 in
theorem outerScan1021 : scan edge (search edge 4) 5 [] 0 tail1021 best = best := by
  have cons : tail1021 = 8161 :: tail1022 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1021_checked]
  change scan edge (search edge 4) 5 [] 0 tail1022
    (if 4 ≤ rootCandidates1021.length then search edge 4 [8161] rootCandidates1021 8161 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1022

set_option maxRecDepth 4096 in
theorem outerScan1020 : scan edge (search edge 4) 5 [] 0 tail1020 best = best := by
  have cons : tail1020 = 8147 :: tail1021 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1020_checked]
  change scan edge (search edge 4) 5 [] 0 tail1021
    (if 4 ≤ rootCandidates1020.length then search edge 4 [8147] rootCandidates1020 8147 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1021

set_option maxRecDepth 4096 in
theorem outerScan1019 : scan edge (search edge 4) 5 [] 0 tail1019 best = best := by
  have cons : tail1019 = 8123 :: tail1020 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1019_checked]
  change scan edge (search edge 4) 5 [] 0 tail1020
    (if 4 ≤ rootCandidates1019.length then search edge 4 [8123] rootCandidates1019 8123 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1020

set_option maxRecDepth 4096 in
theorem outerScan1018 : scan edge (search edge 4) 5 [] 0 tail1018 best = best := by
  have cons : tail1018 = 8117 :: tail1019 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1018_checked]
  change scan edge (search edge 4) 5 [] 0 tail1019
    (if 4 ≤ rootCandidates1018.length then search edge 4 [8117] rootCandidates1018 8117 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1019

set_option maxRecDepth 4096 in
theorem outerScan1017 : scan edge (search edge 4) 5 [] 0 tail1017 best = best := by
  have cons : tail1017 = 8111 :: tail1018 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1017_checked]
  change scan edge (search edge 4) 5 [] 0 tail1018
    (if 4 ≤ rootCandidates1017.length then search edge 4 [8111] rootCandidates1017 8111 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1018

set_option maxRecDepth 4096 in
theorem outerScan1016 : scan edge (search edge 4) 5 [] 0 tail1016 best = best := by
  have cons : tail1016 = 8101 :: tail1017 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1016_checked]
  change scan edge (search edge 4) 5 [] 0 tail1017
    (if 4 ≤ rootCandidates1016.length then search edge 4 [8101] rootCandidates1016 8101 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1017

end ProjectEulerSolutions.PrimePairUpper.Search
