import solvers.«95.py.lean.d».Heap
import solvers.«95.py.lean.d».Processed

/-! One source walk step updates the position dictionary and path before reading the sieve. -/
namespace solvers.«95».walk
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

abbrev program := Solvers.P95Ast.module
abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)

structure Addresses where
  table : Nat
  processed : Nat
  path : Nat
  positions : Nat

structure Addresses.Separated (a : Addresses) : Prop where
  tableProcessed : a.table ≠ a.processed
  tablePath : a.table ≠ a.path
  tablePositions : a.table ≠ a.positions
  processedPath : a.processed ≠ a.path
  processedPositions : a.processed ≠ a.positions
  pathPositions : a.path ≠ a.positions

structure Captured (env : HeapCalls.Env) (a : Addresses) (limit current : Nat) (best : Nat × Nat) : Prop where
  getLimit : env "limit" = some (number limit)
  getTable : env "s" = some (.ref a.table)
  getProcessed : env "processed" = some (.ref a.processed)
  getPath : env "path" = some (.ref a.path)
  getPositions : env "idx" = some (.ref a.positions)
  getCurrent : env "n" = some (number current)
  getLength : env "best_len" = some (number best.1)
  getMinimum : env "best_min" = some (number best.2)

theorem Captured.advance {env : HeapCalls.Env} {a : Addresses} {limit current : Nat} {best : Nat × Nat}
    (captured : Captured env a limit current best) (next : Nat) :
    Captured (setLocal env "n" (number next)) a limit next best where
  getLimit := by simpa [setLocal] using captured.getLimit
  getTable := by simpa [setLocal] using captured.getTable
  getProcessed := by simpa [setLocal] using captured.getProcessed
  getPath := by simpa [setLocal] using captured.getPath
  getPositions := by simpa [setLocal] using captured.getPositions
  getCurrent := by simp [setLocal]
  getLength := by simpa [setLocal] using captured.getLength
  getMinimum := by simpa [setLocal] using captured.getMinimum

structure Found (heap : Mutable.Heap) (a : Addresses) (limit : Nat) (f : Nat → Nat)
    (known : Finset Nat) (path : List Nat) (positions : NaturalDictionary.Entries) : Prop where
  table : heap[a.table]? = some (.list (Sieve.table (limit + 1) f))
  processed : heap[a.processed]? = some (.list (processed.items limit known))
  path : heap[a.path]? = some (.list (path.map number))
  positions : NaturalDictionary.Found heap a.positions positions

def stepHeap (heap : Mutable.Heap) (a : Addresses) (current : Nat)
    (path : List Nat) (positions : NaturalDictionary.Entries) : Mutable.Heap :=
  (heap.set a.positions (.dict (NaturalDictionary.encode (NaturalDictionary.insert positions current path.length)))).set
    a.path (.list ((path ++ [current]).map number))

theorem Found.advance {heap : Mutable.Heap} {a : Addresses} {limit : Nat} {f : Nat → Nat}
    {known : Finset Nat} {path : List Nat} {positions : NaturalDictionary.Entries}
    (found : Found heap a limit f known path positions) (separate : a.Separated) (current : Nat) :
    Found (stepHeap heap a current path positions) a limit f known (path ++ [current])
      (NaturalDictionary.insert positions current path.length) where
  table := by simpa only [stepHeap, List.getElem?_set_ne separate.tablePath.symm,
    List.getElem?_set_ne separate.tablePositions.symm] using found.table
  processed := by simpa only [stepHeap, List.getElem?_set_ne separate.processedPath.symm,
    List.getElem?_set_ne separate.processedPositions.symm] using found.processed
  path := List.getElem?_set_self (by simpa only [List.length_set] using (List.getElem?_eq_some_iff.mp found.path).1)
  positions := by
    change (stepHeap heap a current path positions)[a.positions]? = _
    rw [stepHeap, List.getElem?_set_ne separate.pathPositions]
    exact List.getElem?_set_self (List.getElem?_eq_some_iff.mp found.positions).1

abbrev stepCode : List Stmt :=
  [.assign [.subscript (.name "idx") (.name "n")] (.builtinCall .len [.name "path"]),
   .expr (.call (.attribute (.name "path") "append") [.name "n"]),
   .assign [.name "n"] (.subscript (.name "s") (.name "n"))]

variable {globals : Mutable.Globals}

theorem iteration (heap : Mutable.Heap) (env : HeapCalls.Env) (a : Addresses) (limit current : Nat)
    (f : Nat → Nat) (known : Finset Nat) (path : List Nat) (positions : NaturalDictionary.Entries)
    (best : Nat × Nat) (inside : current ≤ limit) (separate : a.Separated)
    (found : Found heap a limit f known path positions) (captured : Captured env a limit current best) :
    BlockContinues program env stepCode heap (setLocal env "n" (number (f current)))
      (stepHeap heap a current path positions) globals := by
  let indexed := heap.set a.positions
    (.dict (NaturalDictionary.encode (NaturalDictionary.insert positions current path.length)))
  have length : Evaluates program env (.builtinCall .len [.name "path"]) heap
      (number path.length) heap globals := by
    apply Evaluates.builtin (.cons (Evaluates.lookup captured.getPath) .nil)
    simp [primitive, Mutable.length, found.path, number]
  have written := NaturalDictionary.insert_evaluates (globals := globals) (program := program)
    heap env "idx" "n" a.positions current path.length positions (.builtinCall .len [.name "path"])
    found.positions captured.getPositions captured.getCurrent length
  have pathFound : indexed[a.path]? = some (.list (path.map number)) := by
    simpa only [indexed, List.getElem?_set_ne separate.pathPositions.symm] using found.path
  have appended : Evaluates program env (.call (.attribute (.name "path") "append") [.name "n"])
      indexed (.scalar .none) (stepHeap heap a current path positions) globals := by
    have executed := Evaluates.append (globals := globals) (module := program)
      (Evaluates.lookup captured.getPath) pathFound (Evaluates.lookup captured.getCurrent) pathFound
    simpa only [stepHeap, List.map_append, List.map_cons, List.map_nil] using executed
  have nextFound := found.advance separate current
  have next : Evaluates program env (.subscript (.name "s") (.name "n"))
      (stepHeap heap a current path positions) (number (f current)) (stepHeap heap a current path positions) globals := by
    apply Evaluates.subscript (by intros; simp) (Evaluates.lookup captured.getTable) (Evaluates.lookup captured.getCurrent)
    have selected := Sieve.table_lookup (limit + 1) f current (by omega)
    simpa only [(List.getElem?_eq_some_iff.mp selected).2] using
      Mutable.getItem_list (stepHeap heap a current path positions) a.table current _ nextFound.table (by simp; omega)
  exact written.append ((BlockContinues.discard appended).append (BlockContinues.assign next))

end solvers.«95».walk
