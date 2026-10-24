import ProjectEulerSolutions.Lib.CubicPermutations

/-! Ordered count/minimum groups for the cubic-permutation search. -/
namespace ProjectEulerSolutions.CubicGroups
open CubicPermutations
set_option Elab.async false

structure Group where
  digits : List Nat
  count : Int
  minimum : Int
deriving DecidableEq

def fresh (cube : Nat) : Group := ⟨signature cube, 1, cube⟩
def bump (cube : Nat) (group : Group) : Group :=
  { group with count := group.count + 1, minimum := min group.minimum cube }
def update (cube : Nat) (group : Group) : Group :=
  if group.digits = signature cube then bump cube group else group
def contains (cube : Nat) (groups : List Group) : Bool :=
  groups.any (fun group => decide (group.digits = signature cube))
def record (cube : Nat) (groups : List Group) : List Group :=
  if contains cube groups then groups.map (update cube) else groups ++ [fresh cube]
def matching (target : Int) (groups : List Group) : List Int :=
  groups.filterMap (fun group => if group.count = target then some group.minimum else none)

@[simp] theorem update_digits (cube : Nat) (group : Group) : (update cube group).digits = group.digits := by
  by_cases same : group.digits = signature cube <;> simp [update, bump, same]

theorem record_digits (cube : Nat) (groups : List Group) :
    (record cube groups).map Group.digits =
      if contains cube groups then groups.map Group.digits else groups.map Group.digits ++ [signature cube] := by
  unfold record
  split <;> simp [List.map_map, Function.comp_def, fresh]

@[simp] theorem record_nil (cube : Nat) : record cube [] = [fresh cube] := rfl

end ProjectEulerSolutions.CubicGroups
