import «.formalsw».solvers.«17.py.ast»
import lang.py.FormalSW.DictionaryCalls
import ProjectEulerStatements.Lib.NumberWords

namespace solvers.«17»
open FormalSW.PyAst
open ProjectEulerStatements.P17
private abbrev program := Solvers.P17Ast.module

/-- At most ten integer-keyed entries, with the required lookups and word lengths. -/
def WordTable (heap : Mutable.Heap) (globals : Mutable.Globals)
    (name : String) (lower upper : Nat) (length : Nat → Nat) : Prop :=
  ∃ address entries, globals.lookup name = some (.ref address) ∧
    heap[address]? = some (.dict entries) ∧
    entries.length ≤ 10 ∧
    (∀ entry ∈ entries, ∃ key : Int, entry.1 = .int key) ∧
    ∀ n, lower ≤ n → n < upper → ∃ word : String,
      ScalarDict.lookup entries (.int n) = some (.scalar (.string word)) ∧
      word.length = length n

theorem WordTable.append {heap : Mutable.Heap} {globals : Mutable.Globals}
    {name : String} {lower upper : Nat} {length : Nat → Nat}
    (table : WordTable heap globals name lower upper length) (extra : Mutable.Heap) :
    WordTable (heap ++ extra) globals name lower upper length := by
  obtain ⟨address, entries, bound, found, count, integerKeys, words⟩ := table
  refine ⟨address, entries, bound, ?_, count, integerKeys, words⟩
  rw [List.getElem?_append_left]
  · exact found
  · by_contra outside
    simp [List.getElem?_eq_none (show heap.length ≤ address by omega)] at found

theorem WordTable.read {heap : Mutable.Heap} {globals : Mutable.Globals}
    {name : String} {lower upper : Nat} {length : Nat → Nat}
    (table : WordTable heap globals name lower upper length)
    (env : HeapCalls.Env) (index : String) (n : Nat)
    (getTable : env name = globals.lookup name)
    (getIndex : env index = some (.scalar (.int n)))
    (lo : lower ≤ n) (hi : n < upper) :
    HeapCalls.Evaluates program env
      (.builtinCall .len [.subscript (.name name) (.name index)]) heap
      (.scalar (.int (length n))) heap globals := by
  obtain ⟨address, entries, bound, found, count, integerKeys, words⟩ := table
  obtain ⟨word, present, size⟩ := words n lo hi
  apply HeapCalls.Evaluates.builtin (values := [.scalar (.string word)]) (middle := heap)
  · exact .cons (HeapCalls.Evaluates.subscript
      (by intro a b c impossible; cases impossible)
      (.lookup (getTable.trans bound)) (.lookup getIndex)
      (Mutable.getItem_dictionary heap address entries found (.int n) present)) .nil
  · simp [HeapCalls.primitive, Mutable.length, size]

def Tables (heap : Mutable.Heap) (globals : Mutable.Globals) : Prop :=
  WordTable heap globals "ONES" 0 10 smallWordLen ∧
  WordTable heap globals "TEENS" 10 20 smallWordLen ∧
  WordTable heap globals "TENS" 2 10 tensWordLen

theorem Tables.append {heap : Mutable.Heap} {globals : Mutable.Globals}
    (tables : Tables heap globals) (extra : Mutable.Heap) :
    Tables (heap ++ extra) globals :=
  ⟨tables.1.append extra, tables.2.1.append extra, tables.2.2.append extra⟩

theorem bounded (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (index : String) (n lower upper : Int)
    (getN : env index = some (.scalar (.int n))) (lo : lower ≤ n) :
    HeapCalls.Evaluates program env
      (.compare (.constant (.int lower)) [(.ltE, .name index), (.ltE, .constant (.int upper))])
      heap (.scalar (.bool (decide (n ≤ upper)))) heap globals := by
  apply HeapCalls.Evaluates.chain HeapCalls.Evaluates.constant
  apply HeapCalls.Evaluates.Compares.next (compared := .scalar (.bool true))
    (afterComparison := heap) (HeapCalls.Evaluates.lookup getN)
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp, lo]
  · rfl
  · exact HeapCalls.Evaluates.Compares.last .constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])

theorem equal_string_entry (word : String) (value : Mutable.Value) :
    HeapCalls.equalElements [.scalar (.string word)] [value] =
      some (decide (value = .scalar (.string word))) := by
  cases value with
  | ref address => rfl
  | scalar constant =>
      cases constant <;> simp [HeapCalls.equalElements, scalarCompare, eq_comm]
      split <;> simp_all

theorem equal_string_entries (expected : List (Constant × String))
    (entries : List (Constant × Mutable.Value)) :
    HeapCalls.equalEntries (expected.map fun (key, word) => (key, .scalar (.string word))) entries =
      some (expected.all fun (key, word) => decide
        (ScalarDict.lookup entries key = some (.scalar (.string word)))) := by
  induction expected with
  | nil => rfl
  | cons entry rest ih =>
      obtain ⟨key, word⟩ := entry
      cases found : ScalarDict.lookup entries key with
      | none => simp [HeapCalls.equalEntries, found]
      | some value =>
          simp only [List.map_cons, HeapCalls.equalEntries, found, equal_string_entry]
          by_cases same : value = .scalar (.string word)
          · simp [same, ih, found]
          · simp [same, found]


def wordEntries (expected : List (Constant × String)) : List (Constant × Mutable.Value) :=
  expected.map fun (key, word) => (key, .scalar (.string word))

def tableMatches (expected : List (Constant × String))
    (entries : List (Constant × Mutable.Value)) : Bool :=
  decide (expected.length = entries.length) &&
    expected.all fun (key, word) => decide
      (ScalarDict.lookup entries key = some (.scalar (.string word)))

theorem table_test (expected : List (Constant × String))
    (heap : Mutable.Heap) (globals : Mutable.Globals) (env : HeapCalls.Env)
    (literal : Expr) (name : String) (address : Nat) (entries : List (Constant × Mutable.Value))
    (constructed : HeapCalls.Evaluates program env literal heap (.ref heap.length)
      (heap ++ [.dict (wordEntries expected)]) globals)
    (getTable : env name = some (.ref address)) (found : heap[address]? = some (.dict entries)) :
    HeapCalls.Evaluates program env (.compare literal [(.eq, .name name)]) heap
      (.scalar (.bool (tableMatches expected entries))) (heap ++ [.dict (wordEntries expected)]) globals := by
  have inside : address < heap.length := by
    by_contra outside
    simp [List.getElem?_eq_none (show heap.length ≤ address by omega)] at found
  have preserved : (heap ++ [Mutable.Object.dict (wordEntries expected)])[address]? = some (.dict entries) := by
    rw [List.getElem?_append_left inside]
    exact found
  apply HeapCalls.Evaluates.compare constructed (.lookup getTable)
  simp [HeapCalls.compare, preserved, show heap.length ≠ address by omega]
  simp [HeapCalls.equalDictionaries, wordEntries, equal_string_entries, tableMatches]
  split <;> simp_all <;> rfl

def onesExpected : List (Constant × String) :=
  [(.int 0, ""), (.int 1, "one"), (.int 2, "two"), (.int 3, "three"), (.int 4, "four"), (.int 5, "five"), (.int 6, "six"), (.int 7, "seven"), (.int 8, "eight"), (.int 9, "nine")]

theorem ones_test (heap : Mutable.Heap) (globals : Mutable.Globals)
    (env : HeapCalls.Env) (address : Nat) (entries : List (Constant × Mutable.Value))
    (getTable : env "ONES" = some (.ref address)) (found : heap[address]? = some (.dict entries)) :
    HeapCalls.Evaluates program env Solvers.P17Ast.requires_solve_1 heap
      (.scalar (.bool (tableMatches onesExpected entries)))
      (heap ++ [.dict (wordEntries onesExpected)]) globals := by
  apply table_test onesExpected heap globals env _ "ONES" address entries _ getTable found
  exact HeapCalls.Evaluates.dictionary_constants (onesExpected.map fun (key, word) => (key, .string word))

theorem ones_of_condition (heap : Mutable.Heap) (globals : Mutable.Globals)
    (address : Nat) (bound : globals.lookup "ONES" = some (.ref address))
    (typed : Mutable.Value.HasType heap (.ref address) "dict[int,str]")
    (condition : HeapCalls.Condition program (fun name => globals.lookup name)
      Solvers.P17Ast.requires_solve_1 heap globals) :
    WordTable heap globals "ONES" 0 10 smallWordLen := by
  obtain ⟨entries, found, types⟩ := Mutable.Value.dictionary_of_hasType typed rfl
  have executed := ones_test heap globals (fun name => globals.lookup name) address entries bound found
  have matched : tableMatches onesExpected entries = true := by
    simpa using executed.condition_iff.mp condition
  have keys := List.all_eq_true.mp (Bool.and_eq_true_iff.mp matched).2
  have count : entries.length ≤ 10 := by
    have equal : onesExpected.length = entries.length :=
      of_decide_eq_true (Bool.and_eq_true_iff.mp matched).1
    have expectedCount : onesExpected.length = 10 := rfl
    omega
  have integerKeys : ∀ entry ∈ entries, ∃ key : Int, entry.1 = .int key := by
    intro entry member
    have keyType := (types entry member).1
    cases key : entry.1 <;> simp_all [Mutable.scalarKind, Data.typeName]
  refine ⟨address, entries, bound, found, count, integerKeys, ?_⟩
  intro n lo hi
  refine ⟨onesWords[n - 0]!, ?_, ?_⟩
  · exact of_decide_eq_true (keys (.int n, onesWords[n - 0]!) (by interval_cases n <;> simp [onesExpected, onesWords]))
  · simpa using onesWords_length n hi

def teensExpected : List (Constant × String) :=
  [(.int 10, "ten"), (.int 11, "eleven"), (.int 12, "twelve"), (.int 13, "thirteen"), (.int 14, "fourteen"), (.int 15, "fifteen"), (.int 16, "sixteen"), (.int 17, "seventeen"), (.int 18, "eighteen"), (.int 19, "nineteen")]

theorem teens_test (heap : Mutable.Heap) (globals : Mutable.Globals)
    (env : HeapCalls.Env) (address : Nat) (entries : List (Constant × Mutable.Value))
    (getTable : env "TEENS" = some (.ref address)) (found : heap[address]? = some (.dict entries)) :
    HeapCalls.Evaluates program env Solvers.P17Ast.requires_solve_2 heap
      (.scalar (.bool (tableMatches teensExpected entries)))
      (heap ++ [.dict (wordEntries teensExpected)]) globals := by
  apply table_test teensExpected heap globals env _ "TEENS" address entries _ getTable found
  exact HeapCalls.Evaluates.dictionary_constants (teensExpected.map fun (key, word) => (key, .string word))

theorem teens_of_condition (heap : Mutable.Heap) (globals : Mutable.Globals)
    (address : Nat) (bound : globals.lookup "TEENS" = some (.ref address))
    (typed : Mutable.Value.HasType heap (.ref address) "dict[int,str]")
    (condition : HeapCalls.Condition program (fun name => globals.lookup name)
      Solvers.P17Ast.requires_solve_2 heap globals) :
    WordTable heap globals "TEENS" 10 20 smallWordLen := by
  obtain ⟨entries, found, types⟩ := Mutable.Value.dictionary_of_hasType typed rfl
  have executed := teens_test heap globals (fun name => globals.lookup name) address entries bound found
  have matched : tableMatches teensExpected entries = true := by
    simpa using executed.condition_iff.mp condition
  have keys := List.all_eq_true.mp (Bool.and_eq_true_iff.mp matched).2
  have count : entries.length ≤ 10 := by
    have equal : teensExpected.length = entries.length :=
      of_decide_eq_true (Bool.and_eq_true_iff.mp matched).1
    have expectedCount : teensExpected.length = 10 := rfl
    omega
  have integerKeys : ∀ entry ∈ entries, ∃ key : Int, entry.1 = .int key := by
    intro entry member
    have keyType := (types entry member).1
    cases key : entry.1 <;> simp_all [Mutable.scalarKind, Data.typeName]
  refine ⟨address, entries, bound, found, count, integerKeys, ?_⟩
  intro n lo hi
  refine ⟨teenWords[n - 10]!, ?_, ?_⟩
  · exact of_decide_eq_true (keys (.int n, teenWords[n - 10]!) (by interval_cases n <;> simp [teensExpected, teenWords]))
  · simpa using teenWords_length n lo hi

def tensExpected : List (Constant × String) :=
  [(.int 2, "twenty"), (.int 3, "thirty"), (.int 4, "forty"), (.int 5, "fifty"), (.int 6, "sixty"), (.int 7, "seventy"), (.int 8, "eighty"), (.int 9, "ninety")]

theorem tens_test (heap : Mutable.Heap) (globals : Mutable.Globals)
    (env : HeapCalls.Env) (address : Nat) (entries : List (Constant × Mutable.Value))
    (getTable : env "TENS" = some (.ref address)) (found : heap[address]? = some (.dict entries)) :
    HeapCalls.Evaluates program env Solvers.P17Ast.requires_solve_3 heap
      (.scalar (.bool (tableMatches tensExpected entries)))
      (heap ++ [.dict (wordEntries tensExpected)]) globals := by
  apply table_test tensExpected heap globals env _ "TENS" address entries _ getTable found
  exact HeapCalls.Evaluates.dictionary_constants (tensExpected.map fun (key, word) => (key, .string word))

theorem tens_of_condition (heap : Mutable.Heap) (globals : Mutable.Globals)
    (address : Nat) (bound : globals.lookup "TENS" = some (.ref address))
    (typed : Mutable.Value.HasType heap (.ref address) "dict[int,str]")
    (condition : HeapCalls.Condition program (fun name => globals.lookup name)
      Solvers.P17Ast.requires_solve_3 heap globals) :
    WordTable heap globals "TENS" 2 10 tensWordLen := by
  obtain ⟨entries, found, types⟩ := Mutable.Value.dictionary_of_hasType typed rfl
  have executed := tens_test heap globals (fun name => globals.lookup name) address entries bound found
  have matched : tableMatches tensExpected entries = true := by
    simpa using executed.condition_iff.mp condition
  have keys := List.all_eq_true.mp (Bool.and_eq_true_iff.mp matched).2
  have count : entries.length ≤ 10 := by
    have equal : tensExpected.length = entries.length :=
      of_decide_eq_true (Bool.and_eq_true_iff.mp matched).1
    have expectedCount : tensExpected.length = 8 := rfl
    omega
  have integerKeys : ∀ entry ∈ entries, ∃ key : Int, entry.1 = .int key := by
    intro entry member
    have keyType := (types entry member).1
    cases key : entry.1 <;> simp_all [Mutable.scalarKind, Data.typeName]
  refine ⟨address, entries, bound, found, count, integerKeys, ?_⟩
  intro n lo hi
  refine ⟨tensWords[n - 2]!, ?_, ?_⟩
  · exact of_decide_eq_true (keys (.int n, tensWords[n - 2]!) (by interval_cases n <;> simp [tensExpected, tensWords]))
  · simpa using tensWords_length n lo hi

abbrev context (ones teens tens : Nat) : Mutable.Globals :=
  [("ONES", .ref ones), ("TEENS", .ref teens), ("TENS", .ref tens)]

abbrev conditionEnv (n : Int) (ones teens tens : Nat) : HeapCalls.Env :=
  fun name => if name = "n" then some (.scalar (.int n)) else (context ones teens tens).lookup name

abbrev boundCondition (heap : Mutable.Heap) (n : Int) (ones teens tens : Nat) : Prop :=
  HeapCalls.Condition program (conditionEnv n ones teens tens) Solvers.P17Ast.requires_solve_0
    heap (context ones teens tens)

abbrev tableCondition (expression : Expr) (heap : Mutable.Heap) (ones teens tens : Nat) : Prop :=
  HeapCalls.Condition program (fun name => (context ones teens tens).lookup name) expression
    heap (context ones teens tens)

theorem bound_test (heap : Mutable.Heap) (n : Int) (ones teens tens : Nat) :
    HeapCalls.Evaluates program (conditionEnv n ones teens tens) Solvers.P17Ast.requires_solve_0 heap
      (.scalar (.bool (decide (0 ≤ n ∧ n ≤ 1000)))) heap (context ones teens tens) := by
  by_cases nonnegative : 0 ≤ n
  · simpa only [nonnegative, true_and] using
      bounded (conditionEnv n ones teens tens) heap (context ones teens tens) "n" n 0 1000 rfl nonnegative
  · simp only [nonnegative, false_and, decide_false]
    apply HeapCalls.Evaluates.chain .constant
    exact HeapCalls.Evaluates.Compares.stop (.lookup (show conditionEnv n ones teens tens "n" = _ from rfl))
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, nonnegative]) rfl

theorem tables_of_conditions (heap : Mutable.Heap) (ones teens tens : Nat)
    (onesTyped : Mutable.Value.HasType heap (.ref ones) "dict[int,str]")
    (teensTyped : Mutable.Value.HasType heap (.ref teens) "dict[int,str]")
    (tensTyped : Mutable.Value.HasType heap (.ref tens) "dict[int,str]")
    (onesCondition : tableCondition Solvers.P17Ast.requires_solve_1 heap ones teens tens)
    (teensCondition : tableCondition Solvers.P17Ast.requires_solve_2 heap ones teens tens)
    (tensCondition : tableCondition Solvers.P17Ast.requires_solve_3 heap ones teens tens) :
    Tables heap (context ones teens tens) :=
  ⟨ones_of_condition heap _ ones rfl onesTyped onesCondition,
   teens_of_condition heap _ teens rfl teensTyped teensCondition,
   tens_of_condition heap _ tens rfl tensTyped tensCondition⟩

end solvers.«17»
