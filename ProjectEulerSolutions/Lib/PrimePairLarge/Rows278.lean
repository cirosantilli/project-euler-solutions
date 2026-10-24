import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8969_small : List Nat := []
theorem row8969_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8969 row8969_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8969_block_005 : List Nat := [9377, 9539]
def plan8969_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 31), .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 31), (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 11), (.firstFactor 19), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .second, .first, .first, (.secondFactor 29), .first, (.firstFactor 19), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8969_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8969 row8969_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8969_block_005
  · decide
  · rfl

def row8969_block_006 : List Nat := [11489]
def plan8969_block_006 : List Budget.Choice := [(.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 17), .second, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 31), .second, .first, (.firstFactor 7), .first, .first, .first, .first, .second, (.firstFactor 13), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), (.secondFactor 13), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .possible, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 11), .first, .first, .second, .first, (.secondFactor 31), .first, (.firstFactor 23), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 37), (.firstFactor 17), .second, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), .first, .second, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 19), .second, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 37), (.secondFactor 7), .first, .second, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8969_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8969 row8969_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8969_block_006
  · decide
  · rfl

def row8969_block_007 : List Nat := [12959, 14159]
def plan8969_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .second, (.firstFactor 7), (.secondFactor 7), (.firstFactor 37), (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 23), .first, (.firstFactor 17), (.firstFactor 19), .first, (.secondFactor 19), .first, (.secondFactor 7), .possible, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 13), (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.secondFactor 13), (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .possible, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, .first, (.secondFactor 31), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row8969_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8969 row8969_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8969_block_007
  · decide
  · rfl

def row8969_block_008 : List Nat := [14657, 15233, 15413, 15443, 16301, 16421]
def plan8969_block_008 : List Budget.Choice := [.first, .first, .first, (.secondFactor 19), .first, (.firstFactor 31), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 29), .first, .possible, (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 19), (.firstFactor 17), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 29), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 37), .second, .first, .first, .second, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, (.secondFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .possible, .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .possible, .first, .first, .possible, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .second, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 29), .first, (.secondFactor 17), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 23), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 7), (.firstFactor 11), .first, .first, .second, .first, .first, .possible, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .possible, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 17), (.firstFactor 23), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row8969_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8969 row8969_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8969_block_008
  · decide
  · rfl

def row8969_block_009 : List Nat := []
def plan8969_block_009 : List Budget.Choice := [.first, (.secondFactor 11), (.secondFactor 17), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8969_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8969 row8969_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8969_block_009
  · decide
  · rfl

def row8969_block_010 : List Nat := []
theorem row8969_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8969 row8969_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8969_block_011 : List Nat := []
theorem row8969_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8969 row8969_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8969_block_012 : List Nat := []
theorem row8969_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8969 row8969_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8969_block_013 : List Nat := []
theorem row8969_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8969 row8969_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8969_large_left_left : List Nat := row8969_small ++ row8969_block_005
theorem row8969_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8969 row8969_large_left_left :=
  Budget.RowUpperBound.branch row8969_small_covers row8969_block_005_covers

def row8969_large_left_right_right : List Nat := row8969_block_007 ++ row8969_block_008
theorem row8969_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8969 row8969_large_left_right_right :=
  Budget.RowUpperBound.branch row8969_block_007_covers row8969_block_008_covers

def row8969_large_left_right : List Nat := row8969_block_006 ++ row8969_large_left_right_right
theorem row8969_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8969 row8969_large_left_right :=
  Budget.RowUpperBound.branch row8969_block_006_covers row8969_large_left_right_right_covers

def row8969_large_left : List Nat := row8969_large_left_left ++ row8969_large_left_right
theorem row8969_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8969 row8969_large_left :=
  Budget.RowUpperBound.branch row8969_large_left_left_covers row8969_large_left_right_covers

def row8969_large_right_left : List Nat := row8969_block_009 ++ row8969_block_010
theorem row8969_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8969 row8969_large_right_left :=
  Budget.RowUpperBound.branch row8969_block_009_covers row8969_block_010_covers

def row8969_large_right_right_right : List Nat := row8969_block_012 ++ row8969_block_013
theorem row8969_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8969 row8969_large_right_right_right :=
  Budget.RowUpperBound.branch row8969_block_012_covers row8969_block_013_covers

def row8969_large_right_right : List Nat := row8969_block_011 ++ row8969_large_right_right_right
theorem row8969_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8969 row8969_large_right_right :=
  Budget.RowUpperBound.branch row8969_block_011_covers row8969_large_right_right_right_covers

def row8969_large_right : List Nat := row8969_large_right_left ++ row8969_large_right_right
theorem row8969_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8969 row8969_large_right :=
  Budget.RowUpperBound.branch row8969_large_right_left_covers row8969_large_right_right_covers

def row8969_large : List Nat := row8969_large_left ++ row8969_large_right
theorem row8969_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8969 row8969_large :=
  Budget.RowUpperBound.branch row8969_large_left_covers row8969_large_right_covers

def row8969 : List Nat := row8969_large
theorem row8969_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8969 row8969 := row8969_large_covers

def row8971_small : List Nat := []
theorem row8971_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8971 row8971_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8971_block_005 : List Nat := [9187, 9739, 9769, 9787]
def plan8971_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 31), (.secondFactor 17), (.firstFactor 29), (.firstFactor 7), .first, .second, (.firstFactor 11), .first, .first, .second, .possible, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .second, .second, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 29), (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 37), .possible, .first, .first, .first, .possible, .second, .possible, .first, .first, .second, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row8971_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8971 row8971_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8971_block_005
  · decide
  · rfl

def row8971_block_006 : List Nat := [11497, 11779, 11833, 12301]
def plan8971_block_006 : List Budget.Choice := [.first, (.firstFactor 37), .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 13), .first, (.secondFactor 23), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .second, .first, .first, (.firstFactor 31), (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.firstFactor 23), .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), (.secondFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .possible, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 37), (.firstFactor 11), .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 19), .first, .possible, (.firstFactor 13), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.secondFactor 23), (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .possible, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8971_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8971 row8971_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8971_block_006
  · decide
  · rfl

def row8971_block_007 : List Nat := [12889, 12973, 13297, 14341]
def plan8971_block_007 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 31), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .second, .first, (.secondFactor 17), .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), (.secondFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 37), (.firstFactor 7), .possible, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 17), .possible, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 19), (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 19), .possible, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 23), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), .second, (.firstFactor 23), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .possible, .first, .first, .first, (.secondFactor 23), .second, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 19), (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row8971_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8971 row8971_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8971_block_007
  · decide
  · rfl

def row8971_block_008 : List Nat := [15277, 15607, 15787, 15859, 15937, 15991]
def plan8971_block_008 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 29), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 11), .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 13), .possible, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .possible, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 23), (.firstFactor 19), (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 13), .possible, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, (.secondFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8971_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8971 row8971_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8971_block_008
  · decide
  · rfl

def row8971_block_009 : List Nat := []
def plan8971_block_009 : List Budget.Choice := [(.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), .second, .first, (.secondFactor 7), (.firstFactor 17), (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 17), (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8971_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8971 row8971_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8971_block_009
  · decide
  · rfl

def row8971_block_010 : List Nat := []
theorem row8971_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8971 row8971_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8971_block_011 : List Nat := []
theorem row8971_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8971 row8971_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8971_block_012 : List Nat := []
theorem row8971_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8971 row8971_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8971_block_013 : List Nat := []
theorem row8971_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8971 row8971_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8971_large_left_left : List Nat := row8971_small ++ row8971_block_005
theorem row8971_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8971 row8971_large_left_left :=
  Budget.RowUpperBound.branch row8971_small_covers row8971_block_005_covers

def row8971_large_left_right_right : List Nat := row8971_block_007 ++ row8971_block_008
theorem row8971_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8971 row8971_large_left_right_right :=
  Budget.RowUpperBound.branch row8971_block_007_covers row8971_block_008_covers

def row8971_large_left_right : List Nat := row8971_block_006 ++ row8971_large_left_right_right
theorem row8971_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8971 row8971_large_left_right :=
  Budget.RowUpperBound.branch row8971_block_006_covers row8971_large_left_right_right_covers

def row8971_large_left : List Nat := row8971_large_left_left ++ row8971_large_left_right
theorem row8971_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8971 row8971_large_left :=
  Budget.RowUpperBound.branch row8971_large_left_left_covers row8971_large_left_right_covers

def row8971_large_right_left : List Nat := row8971_block_009 ++ row8971_block_010
theorem row8971_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8971 row8971_large_right_left :=
  Budget.RowUpperBound.branch row8971_block_009_covers row8971_block_010_covers

def row8971_large_right_right_right : List Nat := row8971_block_012 ++ row8971_block_013
theorem row8971_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8971 row8971_large_right_right_right :=
  Budget.RowUpperBound.branch row8971_block_012_covers row8971_block_013_covers

def row8971_large_right_right : List Nat := row8971_block_011 ++ row8971_large_right_right_right
theorem row8971_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8971 row8971_large_right_right :=
  Budget.RowUpperBound.branch row8971_block_011_covers row8971_large_right_right_right_covers

def row8971_large_right : List Nat := row8971_large_right_left ++ row8971_large_right_right
theorem row8971_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8971 row8971_large_right :=
  Budget.RowUpperBound.branch row8971_large_right_left_covers row8971_large_right_right_covers

def row8971_large : List Nat := row8971_large_left ++ row8971_large_right
theorem row8971_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8971 row8971_large :=
  Budget.RowUpperBound.branch row8971_large_left_covers row8971_large_right_covers

def row8971 : List Nat := row8971_large
theorem row8971_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8971 row8971 := row8971_large_covers

def row8999_small : List Nat := []
theorem row8999_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8999 row8999_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8999_block_005 : List Nat := [9311, 9521, 9857]
def plan8999_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), (.firstFactor 13), (.secondFactor 37), (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 37), .first, (.firstFactor 31), .possible, .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .second, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 7), .first, .possible, (.firstFactor 13), (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .possible, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .second, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 13), .first, (.secondFactor 7), .second, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 37), (.secondFactor 17), .first, .first]
set_option maxRecDepth 2048 in
theorem row8999_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8999 row8999_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8999_block_005
  · decide
  · rfl

def row8999_block_006 : List Nat := [10601, 10799, 10937, 11273, 11399]
def plan8999_block_006 : List Budget.Choice := [(.secondFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 17), .first, .possible, (.secondFactor 13), (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 11), (.secondFactor 17), .second, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 19), .second, .first, (.secondFactor 17), .first, .second, .first, .first, (.firstFactor 23), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .possible, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 29), .possible, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .second, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 31), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 31), (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), (.firstFactor 17), (.firstFactor 13), .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 17), (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .second]
set_option maxRecDepth 2048 in
theorem row8999_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8999 row8999_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8999_block_006
  · decide
  · rfl

def row8999_block_007 : List Nat := [13457, 13577]
def plan8999_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 29), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 11), .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 29), (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 19), .first, (.secondFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 31), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .second, .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), (.secondFactor 13), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, .second, .possible, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .possible, .first, .first, .second, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 31), (.firstFactor 13), .first, (.firstFactor 29), .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 23), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 29), .first, .first, .first, .second, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8999_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8999 row8999_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8999_block_007
  · decide
  · rfl

def row8999_block_008 : List Nat := [14891, 15551, 15629, 16481]
def plan8999_block_008 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 29), .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 31), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 11), .second, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .possible, (.firstFactor 23), .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .second, .first, (.secondFactor 11), .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .second, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 17), (.secondFactor 11), .first, (.firstFactor 11), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), .second, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 19), .first, .first, .second, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 29), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8999_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8999 row8999_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8999_block_008
  · decide
  · rfl

def row8999_block_009 : List Nat := []
def plan8999_block_009 : List Budget.Choice := [.first, (.secondFactor 23), (.secondFactor 7), (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 17), (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 19), (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 23), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 11), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8999_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8999 row8999_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8999_block_009
  · decide
  · rfl

def row8999_block_010 : List Nat := []
theorem row8999_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8999 row8999_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8999_block_011 : List Nat := []
theorem row8999_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8999 row8999_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8999_block_012 : List Nat := []
theorem row8999_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8999 row8999_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8999_block_013 : List Nat := []
theorem row8999_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8999 row8999_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8999_large_left_left : List Nat := row8999_small ++ row8999_block_005
theorem row8999_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8999 row8999_large_left_left :=
  Budget.RowUpperBound.branch row8999_small_covers row8999_block_005_covers

def row8999_large_left_right_right : List Nat := row8999_block_007 ++ row8999_block_008
theorem row8999_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8999 row8999_large_left_right_right :=
  Budget.RowUpperBound.branch row8999_block_007_covers row8999_block_008_covers

def row8999_large_left_right : List Nat := row8999_block_006 ++ row8999_large_left_right_right
theorem row8999_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8999 row8999_large_left_right :=
  Budget.RowUpperBound.branch row8999_block_006_covers row8999_large_left_right_right_covers

def row8999_large_left : List Nat := row8999_large_left_left ++ row8999_large_left_right
theorem row8999_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8999 row8999_large_left :=
  Budget.RowUpperBound.branch row8999_large_left_left_covers row8999_large_left_right_covers

def row8999_large_right_left : List Nat := row8999_block_009 ++ row8999_block_010
theorem row8999_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8999 row8999_large_right_left :=
  Budget.RowUpperBound.branch row8999_block_009_covers row8999_block_010_covers

def row8999_large_right_right_right : List Nat := row8999_block_012 ++ row8999_block_013
theorem row8999_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8999 row8999_large_right_right_right :=
  Budget.RowUpperBound.branch row8999_block_012_covers row8999_block_013_covers

def row8999_large_right_right : List Nat := row8999_block_011 ++ row8999_large_right_right_right
theorem row8999_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8999 row8999_large_right_right :=
  Budget.RowUpperBound.branch row8999_block_011_covers row8999_large_right_right_right_covers

def row8999_large_right : List Nat := row8999_large_right_left ++ row8999_large_right_right
theorem row8999_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8999 row8999_large_right :=
  Budget.RowUpperBound.branch row8999_large_right_left_covers row8999_large_right_right_covers

def row8999_large : List Nat := row8999_large_left ++ row8999_large_right
theorem row8999_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8999 row8999_large :=
  Budget.RowUpperBound.branch row8999_large_left_covers row8999_large_right_covers

def row8999 : List Nat := row8999_large
theorem row8999_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8999 row8999 := row8999_large_covers

def row9001_small : List Nat := []
theorem row9001_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9001 row9001_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9001_block_005 : List Nat := [9049, 9829, 9859, 9901]
def plan9001_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .possible, .first, (.secondFactor 7), (.secondFactor 31), (.firstFactor 29), (.secondFactor 7), (.firstFactor 11), (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 37), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 29), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 37), .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 31), .second, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 11), .second, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .second, .possible, .first, .first, .first, .first, .possible, .first, .first, .first, .possible, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 29), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row9001_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9001 row9001_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9001_block_005
  · decide
  · rfl

def row9001_block_006 : List Nat := [10723, 10837, 11287, 11353]
def plan9001_block_006 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .possible, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 7), .possible, .first, .first, .first, (.firstFactor 13), .second, .first, .first, (.secondFactor 19), (.firstFactor 29), (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 31), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .second, .first, (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, .possible, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, .possible, .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 19), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .second, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 29), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 17), (.secondFactor 17), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row9001_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9001 row9001_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9001_block_006
  · decide
  · rfl

def row9001_block_007 : List Nat := [12781, 12841, 12853, 13633, 14029]
def plan9001_block_007 : List Budget.Choice := [(.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 19), .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 23), .first, (.secondFactor 11), (.firstFactor 11), .possible, .first, (.secondFactor 13), .first, .first, (.secondFactor 11), (.firstFactor 7), .possible, .possible, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 13), .first, .first, .first, .second, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 17), (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .possible, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .second, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 17), .first, (.secondFactor 17), (.secondFactor 13), .first, (.firstFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row9001_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9001 row9001_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9001_block_007
  · decide
  · rfl

def row9001_block_008 : List Nat := [15313]
def plan9001_block_008 : List Budget.Choice := [(.firstFactor 11), .first, (.firstFactor 37), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 23), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 7), .possible, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), (.secondFactor 13), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 11), .first, .first, .first, .second, .second, (.firstFactor 31), (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 31), (.secondFactor 11), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 19), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 13), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9001_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9001 row9001_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9001_block_008
  · decide
  · rfl

def row9001_block_009 : List Nat := [16831]
def plan9001_block_009 : List Budget.Choice := [(.secondFactor 11), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 17), (.secondFactor 23), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .second, (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 29), (.secondFactor 13), .first, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9001_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9001 row9001_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9001_block_009
  · decide
  · rfl

def row9001_block_010 : List Nat := []
theorem row9001_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9001 row9001_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9001_block_011 : List Nat := []
theorem row9001_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9001 row9001_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9001_block_012 : List Nat := []
theorem row9001_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9001 row9001_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9001_block_013 : List Nat := []
theorem row9001_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9001 row9001_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9001_large_left_left : List Nat := row9001_small ++ row9001_block_005
theorem row9001_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9001 row9001_large_left_left :=
  Budget.RowUpperBound.branch row9001_small_covers row9001_block_005_covers

def row9001_large_left_right_right : List Nat := row9001_block_007 ++ row9001_block_008
theorem row9001_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9001 row9001_large_left_right_right :=
  Budget.RowUpperBound.branch row9001_block_007_covers row9001_block_008_covers

def row9001_large_left_right : List Nat := row9001_block_006 ++ row9001_large_left_right_right
theorem row9001_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9001 row9001_large_left_right :=
  Budget.RowUpperBound.branch row9001_block_006_covers row9001_large_left_right_right_covers

def row9001_large_left : List Nat := row9001_large_left_left ++ row9001_large_left_right
theorem row9001_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9001 row9001_large_left :=
  Budget.RowUpperBound.branch row9001_large_left_left_covers row9001_large_left_right_covers

def row9001_large_right_left : List Nat := row9001_block_009 ++ row9001_block_010
theorem row9001_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9001 row9001_large_right_left :=
  Budget.RowUpperBound.branch row9001_block_009_covers row9001_block_010_covers

def row9001_large_right_right_right : List Nat := row9001_block_012 ++ row9001_block_013
theorem row9001_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9001 row9001_large_right_right_right :=
  Budget.RowUpperBound.branch row9001_block_012_covers row9001_block_013_covers

def row9001_large_right_right : List Nat := row9001_block_011 ++ row9001_large_right_right_right
theorem row9001_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9001 row9001_large_right_right :=
  Budget.RowUpperBound.branch row9001_block_011_covers row9001_large_right_right_right_covers

def row9001_large_right : List Nat := row9001_large_right_left ++ row9001_large_right_right
theorem row9001_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9001 row9001_large_right :=
  Budget.RowUpperBound.branch row9001_large_right_left_covers row9001_large_right_right_covers

def row9001_large : List Nat := row9001_large_left ++ row9001_large_right
theorem row9001_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9001 row9001_large :=
  Budget.RowUpperBound.branch row9001_large_left_covers row9001_large_right_covers

def row9001 : List Nat := row9001_large
theorem row9001_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9001 row9001 := row9001_large_covers

end ProjectEulerSolutions.PrimePairLarge
