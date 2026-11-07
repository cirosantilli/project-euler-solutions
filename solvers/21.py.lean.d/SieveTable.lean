import lang.py.FormalSW.HeapCalls

namespace solvers.«21».Sieve
open FormalSW.PyAst

/-- The source list representing the current mathematical sieve values. -/
def table (size : Nat) (value : Nat → Nat) : List Mutable.Value :=
  (List.range size).map (fun n => .scalar (.int (value n)))

@[simp] theorem table_length (size : Nat) (value : Nat → Nat) : (table size value).length = size := by
  simp [table]

theorem table_lookup (size : Nat) (value : Nat → Nat) (position : Nat) (inside : position < size) :
    (table size value)[position]? = some (.scalar (.int (value position))) := by
  simp [table, inside]

end solvers.«21».Sieve
