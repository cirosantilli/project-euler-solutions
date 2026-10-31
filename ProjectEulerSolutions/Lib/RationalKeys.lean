import ProjectEulerSolutions.Lib.RationalHeap
import lang.py.FormalSW.KeyRelocation

/-! Immutable tuple keys for rational fragments and complete reduction states. -/
namespace ProjectEulerSolutions.RationalPair.Heap
open FormalSW.PyAst Mutable.Relocation
set_option Elab.async false

abbrev key (pair : Pair) : HashKey := .tuple [.scalar (.int pair.1), .scalar (.int pair.2)]
abbrev stateKey (pairs : List Pair) : HashKey := .tuple (pairs.map key)

theorem Row.resolve {heap : Mutable.Heap} {value : Mutable.Value} {pair : Pair}
    (row : Row heap value pair) (fuel : Nat) :
    Mutable.keyOfFuel heap (fuel + 1) value = some (.ok (key pair)) := by
  obtain ⟨address, rfl, found⟩ := row
  rw [keyOfFuel_ref, found]
  simp [number, collectResolved, Mutable.keyOfFuel, key, Except.map]

theorem Rows.resolve {heap : Mutable.Heap} {values : List Mutable.Value} {pairs : List Pair}
    (rows : Rows heap values pairs) (fuel : Nat) :
    collectResolved (Mutable.keyOfFuel heap (fuel + 1)) values = some (.ok (pairs.map key)) := by
  induction rows with
  | nil => rfl
  | cons row rows ih => simp [collectResolved, row.resolve fuel, ih, Except.map]

/-- Sharing tuple objects does not affect the value of the state key. -/
theorem state_hash (heap : Mutable.Heap) (address : Nat) (values : List Mutable.Value) (pairs : List Pair)
    (found : heap[address]? = some (.tuple values)) (rows : Rows heap values pairs) :
    Mutable.hashKey (.ref address) heap = some (.ok (stateKey pairs), heap) := by
  have nonempty : 0 < heap.length := Nat.zero_lt_of_lt (List.getElem?_eq_some_iff.mp found).1
  cases size : heap.length with
  | zero => omega
  | succ fuel =>
    simp [Mutable.hashKey, Mutable.keyOf, size, keyOfFuel_ref, found, rows.resolve fuel, stateKey, Except.map]

theorem Row.hash {heap : Mutable.Heap} {value : Mutable.Value} {pair : Pair}
    (row : Row heap value pair) :
    Mutable.hashKey value heap = some (.ok (key pair), heap) := by
  simp [Mutable.hashKey, Mutable.keyOf, row.resolve heap.length]

theorem key_equalFuel (left right : Pair) (fuel : Nat) :
    HashKey.equalFuel (fuel + 2) (key left) (key right) = decide (left = right) := by
  simp [HashKey.equalFuel, List.isEqv, scalarCompare, evalCmpOp, Prod.ext_iff, Bool.decide_and]

theorem keys_equalFuel (left right : List Pair) (fuel : Nat) :
    (left.map key).isEqv (right.map key) (HashKey.equalFuel (fuel + 2)) = decide (left = right) := by
  induction left generalizing right with
  | nil => cases right <;> simp
  | cons pair rest ih =>
    cases right with
    | nil => simp
    | cons other tail => simp [List.isEqv, key_equalFuel, ih, Bool.decide_and]

private theorem keys_depth (pairs : List Pair) : HashKey.listDepth (pairs.map key) = if pairs = [] then 0 else 2 := by
  induction pairs with
  | nil => rfl
  | cons pair rest ih =>
    change max 2 (HashKey.listDepth (rest.map key)) = 2
    rw [ih]
    split_ifs <;> rfl

/-- Tuple memo keys distinguish exactly the ordered sequences of rational pairs. -/
theorem stateKey_equal (left right : List Pair) :
    HashKey.equal (stateKey left) (stateKey right) = decide (left = right) := by
  cases left with
  | nil => cases right <;> simp [HashKey.equal, HashKey.depth, HashKey.equalFuel]
  | cons pair rest =>
    cases right with
    | nil => simp [HashKey.equal, HashKey.depth, HashKey.equalFuel]
    | cons other tail =>
      change HashKey.equalFuel ((HashKey.listDepth ((pair :: rest).map key) + 1) +
        (HashKey.listDepth ((other :: tail).map key) + 1) + 1) _ _ = _
      simp only [keys_depth, List.cons_ne_nil, if_false]
      change (((pair :: rest).map key).isEqv ((other :: tail).map key) (HashKey.equalFuel (4 + 2))) = _
      exact keys_equalFuel _ _ 4

end ProjectEulerSolutions.RationalPair.Heap
