import «.formalsw».solvers.«98.py.ast»
import lang.py.FormalSW.DictionaryCalls
import ProjectEulerSolutions.Lib.AnagramAssignment
import ProjectEulerSolutions.Lib.HeapCollection

namespace solvers.«98».pattern_signature
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.AnagramSquares
set_option Elab.async false

abbrev program := Solvers.P98Ast.module
abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
abbrev character (c : Char) : Mutable.Value := .scalar (.string (String.singleton c))

@[simp] theorem singleton_eq (a b : Char) : String.singleton a = String.singleton b ↔ a = b := by
  constructor
  · intro equal
    have := congrArg String.toList equal
    simpa using this
  · exact congrArg String.singleton

def table : List Char → Nat → List (Constant × Mutable.Value)
  | [], _ => []
  | c :: rest, first => (.string (String.singleton c), number first) :: table rest (first + 1)

theorem lookup_table (known : List Char) (first : Nat) (c : Char) :
    ScalarDict.lookup (table known first) (.string (String.singleton c)) =
      if c ∈ known then some (number (first + known.idxOf c)) else none := by
  induction known generalizing first with
  | nil => rfl
  | cons head tail ih =>
      by_cases same : head = c
      · subst head
        simp [table, ScalarDict.lookup, scalarCompare]
      · simp [table, ScalarDict.lookup, scalarCompare, same, Ne.symm same, ih,
          Nat.add_comm, Nat.add_left_comm]

theorem insert_table (known : List Char) (first : Nat) (c : Char) (fresh : c ∉ known) :
    ScalarDict.insert (table known first) (.string (String.singleton c)) (number (first + known.length)) =
      table (known ++ [c]) first := by
  induction known generalizing first with
  | nil => simp [table, ScalarDict.insert]
  | cons head tail ih =>
      have different : head ≠ c := by intro same; exact fresh (by simp [same])
      have missing : c ∉ tail := by intro member; exact fresh (by simp [member])
      simpa [table, ScalarDict.insert, scalarCompare, different, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]
        using congrArg (List.cons (.string (String.singleton head), number first)) (ih (first + 1) missing)

def work (base : Mutable.Heap) (known : List Char) (output : List Nat) : Mutable.Heap :=
  base ++ [.dict (table known 0), .list (output.map number)]

@[simp] theorem found_table (base : Mutable.Heap) (known : List Char) (output : List Nat) :
    (work base known output)[base.length]? = some (.dict (table known 0)) := by simp [work]

@[simp] theorem found_output (base : Mutable.Heap) (known : List Char) (output : List Nat) :
    (work base known output)[base.length + 1]? = some (.list (output.map number)) := by simp [work]

structure Locals (env : HeapCalls.Env) (base : Mutable.Heap) (known : List Char) : Prop where
  getTable : env "mp" = some (.ref base.length)
  getOutput : env "out" = some (.ref (base.length + 1))
  getNext : env "nxt" = some (number known.length)

theorem membership (base : Mutable.Heap) (known : List Char) (output : List Nat) (c : Char)
    (env : HeapCalls.Env) (locals : Locals env base known) (getC : env "ch" = some (character c)) :
    Evaluates program env (.membership (.name "ch") (.name "mp") true) (work base known output)
      (.scalar (.bool (decide (c ∉ known)))) (work base known output) := by
  apply Evaluates.membership (.cons (Evaluates.lookup getC) (.cons (Evaluates.lookup locals.getTable) .nil))
  by_cases member : c ∈ known <;> simp [HeapCalls.membership, lookup_table, member]

theorem lookup (base : Mutable.Heap) (known : List Char) (output : List Nat) (c : Char) (member : c ∈ known)
    (env : HeapCalls.Env) (locals : Locals env base known) (getC : env "ch" = some (character c)) :
    Evaluates program env (.subscript (.name "mp") (.name "ch")) (work base known output)
      (number (known.idxOf c)) (work base known output) := by
  apply Evaluates.subscript (by intros; simp) (Evaluates.lookup locals.getTable) (Evaluates.lookup getC)
  simp [Mutable.getItem, lookup_table, member]

abbrev recordCode : List Stmt :=
  [.assign [.subscript (.name "mp") (.name "ch")] (.name "nxt"),
   .augAssign (.name "nxt") .add (.constant (.int 1))]

theorem record (base : Mutable.Heap) (known : List Char) (output : List Nat) (c : Char) (fresh : c ∉ known)
    (env : HeapCalls.Env) (locals : Locals env base known) (getC : env "ch" = some (character c)) :
    BlockContinues program env recordCode (work base known output)
      (setLocal env "nxt" (number (known.length + 1))) (work base (known ++ [c]) output) := by
  have written : BlockContinues program env
      [.assign [.subscript (.name "mp") (.name "ch")] (.name "nxt")]
      (work base known output) env (work base (known ++ [c]) output) := by
    apply BlockContinues.setItem (by intros; simp) (Evaluates.lookup locals.getNext)
      (.cons (Evaluates.lookup locals.getTable) (.cons (Evaluates.lookup getC) .nil))
    have inserted := insert_table known 0 c fresh
    simp only [Nat.zero_add] at inserted
    simp [Mutable.setItem, inserted, work, List.set_append_right, Except.map]
  apply written.append
  apply BlockContinues.augAssign locals.getNext Evaluates.constant
  simp [inplace, Mutable.binary, scalarBinary, number]

theorem append (base : Mutable.Heap) (known : List Char) (output : List Nat) (c : Char) (member : c ∈ known)
    (env : HeapCalls.Env) (locals : Locals env base known) (getC : env "ch" = some (character c)) :
    Evaluates program env (.call (.attribute (.name "out") "append") [.subscript (.name "mp") (.name "ch")])
      (work base known output) (.scalar .none) (work base known (output ++ [known.idxOf c])) := by
  have executed := Evaluates.append (module := program) (Evaluates.lookup locals.getOutput)
    (found_output base known output) (lookup base known output c member env locals getC) (found_output base known output)
  simpa [work, List.map_append, List.set_append_right] using executed

end solvers.«98».pattern_signature
