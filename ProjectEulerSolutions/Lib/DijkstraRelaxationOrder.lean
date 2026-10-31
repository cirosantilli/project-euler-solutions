import ProjectEulerSolutions.Lib.DijkstraArrayRelaxation
import Mathlib.Data.List.Perm.Basic

/-! Relaxations from a fixed source commute, including repeated targets. -/
namespace ProjectEulerSolutions.DijkstraArrayRelaxation
open ProjectEulerSolutions.P83

theorem relax_comm (matrix : List (List Int)) (width : Nat) (visited : Array Bool)
    (distance : Int) (distances : Array Int) (a b : Int × Int) :
    relax matrix width visited distance (relax matrix width visited distance distances a) b =
      relax matrix width visited distance (relax matrix width visited distance distances b) a := by
  apply Array.ext (by simp only [relax_size])
  intro i leftInside rightInside
  have inside : i < distances.size := by simpa only [relax_size] using leftInside
  have leftRead := relax_read matrix width visited distance
    (relax matrix width visited distance distances a) b i (by simpa only [relax_size] using inside)
  have rightRead := relax_read matrix width visited distance
    (relax matrix width visited distance distances b) a i (by simpa only [relax_size] using inside)
  rw [relax_read matrix width visited distance distances a i inside] at leftRead
  rw [relax_read matrix width visited distance distances b i inside] at rightRead
  simp only [getElem!_pos, leftInside] at leftRead
  simp only [getElem!_pos, rightInside] at rightRead
  rw [leftRead, rightRead]
  split_ifs <;> omega

theorem fold_perm (matrix : List (List Int)) (width : Nat) (visited : Array Bool)
    (distance : Int) (distances : Array Int) (cells other : List (Int × Int))
    (permutation : cells.Perm other) :
    cells.foldl (relax matrix width visited distance) distances =
      other.foldl (relax matrix width visited distance) distances := by
  exact permutation.foldl_eq' (fun a _ b _ values => relax_comm matrix width visited distance values a b) distances

end ProjectEulerSolutions.DijkstraArrayRelaxation
