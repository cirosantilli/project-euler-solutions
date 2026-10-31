import solvers.«89.py.lean.d».Equivalence

namespace solvers.«89».source_equivalence
open FormalSW.PyAst ProjectEulerSolutions
open int_to_min_roman
set_option Elab.async false
set_option maxHeartbeats 1000000

/-- One concrete representation of the standard tables, independent of the input list's size. -/
def standardEntries : List Entry :=
  RomanNumerals.tokens.zipIdx.map (fun (pair, index) => ⟨pair.1, pair.2, index + 1⟩)

def standardBase : Mutable.Heap :=
  [.dict data.«project-euler-statements».data.py.project_euler_statements.p89.valueEntries] ++
    standardEntries.map (fun entry => .tuple [number entry.denomination, textValue entry.token]) ++
    [.tuple (standardEntries.map Entry.reference)]

def standardGlobals : Mutable.Globals :=
  [("ROMAN_VALUE", .ref 0), ("MINIMAL_TABLE", .ref (standardEntries.length + 1))]

def standardHeap (lines : List String) : Mutable.Heap := standardBase ++ [.list (lines.map textValue)]

theorem standard_symbols : SymbolTable standardBase standardGlobals := by
  refine ⟨0, data.«project-euler-statements».data.py.project_euler_statements.p89.valueEntries, rfl, rfl, ?_⟩
  exact data.«project-euler-statements».data.py.project_euler_statements.p89.value_lookup

theorem standard_tokens : EncodingTable standardBase standardGlobals := by
  refine ⟨standardEntries.length + 1, standardEntries, rfl, ?_, ?_, ?_⟩
  · rfl
  · intro entry member
    simp only [standardEntries, RomanNumerals.tokens, List.zipIdx_cons, List.zipIdx_nil,
      List.map_cons, List.map_nil, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      constructor <;> decide
  · rfl

/-- Every valid input list, with no length or value bound, has a valid public entry state. -/
theorem standard_domain (lines : List String) (valid : ∀ text ∈ lines, ValidText text) :
    Domain (standardHeap lines) standardGlobals standardBase.length lines := by
  refine ⟨?_, ?_, ?_, ?_, valid, standard_symbols.append _, standard_tokens.append _⟩
  · simp [Mutable.Heap.Valid, standardHeap, standardBase, standardEntries, RomanNumerals.tokens,
      data.«project-euler-statements».data.py.project_euler_statements.p89.valueEntries,
      data.«project-euler-statements».data.py.project_euler_statements.p89.valueTable,
      ScalarDict.insert, ScalarDict.Valid, scalarCompare, Entry.reference]
    constructor
    · simp [ScalarSet.Valid, scalarCompare]
    · intro key value member
      rcases member with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ |
        ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> trivial
  · simp [standardGlobals]
  · intro entry member
    simp only [standardGlobals, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · change 0 < standardBase.length + 1
      decide
    · change standardEntries.length + 1 < standardBase.length + 1
      decide
  · simp [standardHeap]

end solvers.«89».source_equivalence
