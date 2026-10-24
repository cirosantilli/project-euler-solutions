import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row10427_small : List Nat := []
theorem row10427_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10427 row10427_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10427_block_005 : List Nat := []
theorem row10427_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10427 row10427_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10427_block_006 : List Nat := [10667, 12401]
def plan10427_block_006 : List Budget.Choice := [.outside, .outside, .outside, .first, .second, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, .possible, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 29), .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 23), (.secondFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .second, (.secondFactor 7), .first, (.secondFactor 19), (.secondFactor 17), (.firstFactor 31), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 17), (.firstFactor 11), (.secondFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 13), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row10427_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10427 row10427_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10427_block_006
  · decide
  · rfl

def row10427_block_007 : List Nat := [12689, 13229, 13781]
def plan10427_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 23), (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 23), (.firstFactor 7), .possible, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .second, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .second, .first, (.secondFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 23), (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 23), .second, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.secondFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .possible, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .second, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 23), .first, (.secondFactor 19), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .second, (.firstFactor 11), .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 23), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row10427_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10427 row10427_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10427_block_007
  · decide
  · rfl

def row10427_block_008 : List Nat := [14627, 14753]
def plan10427_block_008 : List Budget.Choice := [.first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 13), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 13), .possible, .first, (.firstFactor 29), (.secondFactor 7), .first, .second, (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 37), .first, .first, .first, .first, .possible, (.secondFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .second, .second, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .second, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .second, (.firstFactor 19), .first, .first, (.firstFactor 31), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 31), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10427_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10427 row10427_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10427_block_008
  · decide
  · rfl

def row10427_block_009 : List Nat := []
theorem row10427_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10427 row10427_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10427_block_010 : List Nat := []
theorem row10427_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10427 row10427_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10427_block_011 : List Nat := []
theorem row10427_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10427 row10427_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10427_block_012 : List Nat := []
theorem row10427_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10427 row10427_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10427_block_013 : List Nat := []
theorem row10427_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10427 row10427_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10427_large_left_left : List Nat := row10427_small ++ row10427_block_005
theorem row10427_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10427 row10427_large_left_left :=
  Budget.RowUpperBound.branch row10427_small_covers row10427_block_005_covers

def row10427_large_left_right_right : List Nat := row10427_block_007 ++ row10427_block_008
theorem row10427_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10427 row10427_large_left_right_right :=
  Budget.RowUpperBound.branch row10427_block_007_covers row10427_block_008_covers

def row10427_large_left_right : List Nat := row10427_block_006 ++ row10427_large_left_right_right
theorem row10427_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10427 row10427_large_left_right :=
  Budget.RowUpperBound.branch row10427_block_006_covers row10427_large_left_right_right_covers

def row10427_large_left : List Nat := row10427_large_left_left ++ row10427_large_left_right
theorem row10427_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10427 row10427_large_left :=
  Budget.RowUpperBound.branch row10427_large_left_left_covers row10427_large_left_right_covers

def row10427_large_right_left : List Nat := row10427_block_009 ++ row10427_block_010
theorem row10427_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10427 row10427_large_right_left :=
  Budget.RowUpperBound.branch row10427_block_009_covers row10427_block_010_covers

def row10427_large_right_right_right : List Nat := row10427_block_012 ++ row10427_block_013
theorem row10427_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10427 row10427_large_right_right_right :=
  Budget.RowUpperBound.branch row10427_block_012_covers row10427_block_013_covers

def row10427_large_right_right : List Nat := row10427_block_011 ++ row10427_large_right_right_right
theorem row10427_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10427 row10427_large_right_right :=
  Budget.RowUpperBound.branch row10427_block_011_covers row10427_large_right_right_right_covers

def row10427_large_right : List Nat := row10427_large_right_left ++ row10427_large_right_right
theorem row10427_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10427 row10427_large_right :=
  Budget.RowUpperBound.branch row10427_large_right_left_covers row10427_large_right_right_covers

def row10427_large : List Nat := row10427_large_left ++ row10427_large_right
theorem row10427_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10427 row10427_large :=
  Budget.RowUpperBound.branch row10427_large_left_covers row10427_large_right_covers

def row10427 : List Nat := row10427_large
theorem row10427_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10427 row10427 := row10427_large_covers

def row10429_small : List Nat := []
theorem row10429_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10429 row10429_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10429_block_005 : List Nat := []
theorem row10429_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10429 row10429_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10429_block_006 : List Nat := [11503, 11887]
def plan10429_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 19), .first, (.firstFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.secondFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .second, (.secondFactor 19), .first, .first, .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 37), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .second, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .possible, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .second, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .second, .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 29), (.firstFactor 23), (.firstFactor 7), .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 29), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .second, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), (.firstFactor 23), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row10429_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10429 row10429_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10429_block_006
  · decide
  · rfl

def row10429_block_007 : List Nat := [12511, 13381, 13537, 14083]
def plan10429_block_007 : List Budget.Choice := [(.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 7), .second, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .second, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 29), (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 29), (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 13), .first, .possible, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 31), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, (.secondFactor 17), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 37), .first, (.firstFactor 17), (.secondFactor 19), .first, .first, .first, (.secondFactor 31), .first, .possible, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 37), .first, .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 11), .second, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 37), (.firstFactor 23)]
set_option maxRecDepth 2048 in
theorem row10429_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10429 row10429_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10429_block_007
  · decide
  · rfl

def row10429_block_008 : List Nat := [14683, 14797]
def plan10429_block_008 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 11), .first, .first, .possible, .first, (.secondFactor 19), .first, .first, (.firstFactor 19), (.firstFactor 23), .first, .first, .first, .first, .second, .first, .second, .first, .possible, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 19), .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 37), (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10429_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10429 row10429_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10429_block_008
  · decide
  · rfl

def row10429_block_009 : List Nat := []
theorem row10429_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10429 row10429_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10429_block_010 : List Nat := []
theorem row10429_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10429 row10429_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10429_block_011 : List Nat := []
theorem row10429_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10429 row10429_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10429_block_012 : List Nat := []
theorem row10429_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10429 row10429_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10429_block_013 : List Nat := []
theorem row10429_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10429 row10429_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10429_large_left_left : List Nat := row10429_small ++ row10429_block_005
theorem row10429_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10429 row10429_large_left_left :=
  Budget.RowUpperBound.branch row10429_small_covers row10429_block_005_covers

def row10429_large_left_right_right : List Nat := row10429_block_007 ++ row10429_block_008
theorem row10429_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10429 row10429_large_left_right_right :=
  Budget.RowUpperBound.branch row10429_block_007_covers row10429_block_008_covers

def row10429_large_left_right : List Nat := row10429_block_006 ++ row10429_large_left_right_right
theorem row10429_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10429 row10429_large_left_right :=
  Budget.RowUpperBound.branch row10429_block_006_covers row10429_large_left_right_right_covers

def row10429_large_left : List Nat := row10429_large_left_left ++ row10429_large_left_right
theorem row10429_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10429 row10429_large_left :=
  Budget.RowUpperBound.branch row10429_large_left_left_covers row10429_large_left_right_covers

def row10429_large_right_left : List Nat := row10429_block_009 ++ row10429_block_010
theorem row10429_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10429 row10429_large_right_left :=
  Budget.RowUpperBound.branch row10429_block_009_covers row10429_block_010_covers

def row10429_large_right_right_right : List Nat := row10429_block_012 ++ row10429_block_013
theorem row10429_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10429 row10429_large_right_right_right :=
  Budget.RowUpperBound.branch row10429_block_012_covers row10429_block_013_covers

def row10429_large_right_right : List Nat := row10429_block_011 ++ row10429_large_right_right_right
theorem row10429_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10429 row10429_large_right_right :=
  Budget.RowUpperBound.branch row10429_block_011_covers row10429_large_right_right_right_covers

def row10429_large_right : List Nat := row10429_large_right_left ++ row10429_large_right_right
theorem row10429_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10429 row10429_large_right :=
  Budget.RowUpperBound.branch row10429_large_right_left_covers row10429_large_right_right_covers

def row10429_large : List Nat := row10429_large_left ++ row10429_large_right
theorem row10429_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10429 row10429_large :=
  Budget.RowUpperBound.branch row10429_large_left_covers row10429_large_right_covers

def row10429 : List Nat := row10429_large
theorem row10429_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10429 row10429 := row10429_large_covers

def row10433_small : List Nat := []
theorem row10433_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10433 row10433_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10433_block_005 : List Nat := []
theorem row10433_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10433 row10433_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10433_block_006 : List Nat := [10781, 12041]
def plan10433_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .second, .first, .first, .possible, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .second, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 19), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, (.firstFactor 17), .second, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .second, .first, (.firstFactor 37), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 37), (.secondFactor 29), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 23), (.secondFactor 17), (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .second, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, (.firstFactor 19), .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row10433_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10433 row10433_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10433_block_006
  · decide
  · rfl

def row10433_block_007 : List Nat := [12713, 13709, 13841, 13967]
def plan10433_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 29), .first, .first, .first, (.secondFactor 13), .second, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .second, (.firstFactor 29), (.secondFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .second, (.secondFactor 13), .first, (.secondFactor 37), (.firstFactor 11), (.secondFactor 29), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 31), (.secondFactor 29), .first, (.secondFactor 7), .first, .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 11), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 31), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .possible, .first, .second, .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .possible, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .possible, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 13), (.secondFactor 17), .first, .second, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row10433_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10433 row10433_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10433_block_007
  · decide
  · rfl

def row10433_block_008 : List Nat := [14549]
def plan10433_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .second, .possible, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .second, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 7), .first, .second, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), (.firstFactor 29), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 29), .first, .second, (.firstFactor 19), .first, (.secondFactor 23), (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, .second, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 13), (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10433_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10433 row10433_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10433_block_008
  · decide
  · rfl

def row10433_block_009 : List Nat := []
theorem row10433_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10433 row10433_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10433_block_010 : List Nat := []
theorem row10433_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10433 row10433_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10433_block_011 : List Nat := []
theorem row10433_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10433 row10433_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10433_block_012 : List Nat := []
theorem row10433_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10433 row10433_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10433_block_013 : List Nat := []
theorem row10433_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10433 row10433_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10433_large_left_left : List Nat := row10433_small ++ row10433_block_005
theorem row10433_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10433 row10433_large_left_left :=
  Budget.RowUpperBound.branch row10433_small_covers row10433_block_005_covers

def row10433_large_left_right_right : List Nat := row10433_block_007 ++ row10433_block_008
theorem row10433_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10433 row10433_large_left_right_right :=
  Budget.RowUpperBound.branch row10433_block_007_covers row10433_block_008_covers

def row10433_large_left_right : List Nat := row10433_block_006 ++ row10433_large_left_right_right
theorem row10433_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10433 row10433_large_left_right :=
  Budget.RowUpperBound.branch row10433_block_006_covers row10433_large_left_right_right_covers

def row10433_large_left : List Nat := row10433_large_left_left ++ row10433_large_left_right
theorem row10433_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10433 row10433_large_left :=
  Budget.RowUpperBound.branch row10433_large_left_left_covers row10433_large_left_right_covers

def row10433_large_right_left : List Nat := row10433_block_009 ++ row10433_block_010
theorem row10433_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10433 row10433_large_right_left :=
  Budget.RowUpperBound.branch row10433_block_009_covers row10433_block_010_covers

def row10433_large_right_right_right : List Nat := row10433_block_012 ++ row10433_block_013
theorem row10433_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10433 row10433_large_right_right_right :=
  Budget.RowUpperBound.branch row10433_block_012_covers row10433_block_013_covers

def row10433_large_right_right : List Nat := row10433_block_011 ++ row10433_large_right_right_right
theorem row10433_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10433 row10433_large_right_right :=
  Budget.RowUpperBound.branch row10433_block_011_covers row10433_large_right_right_right_covers

def row10433_large_right : List Nat := row10433_large_right_left ++ row10433_large_right_right
theorem row10433_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10433 row10433_large_right :=
  Budget.RowUpperBound.branch row10433_large_right_left_covers row10433_large_right_right_covers

def row10433_large : List Nat := row10433_large_left ++ row10433_large_right
theorem row10433_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10433 row10433_large :=
  Budget.RowUpperBound.branch row10433_large_left_covers row10433_large_right_covers

def row10433 : List Nat := row10433_large
theorem row10433_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10433 row10433 := row10433_large_covers

def row10453_small : List Nat := []
theorem row10453_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10453 row10453_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10453_block_005 : List Nat := []
theorem row10453_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10453 row10453_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10453_block_006 : List Nat := [10477, 11329, 11503]
def plan10453_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 29), .first, (.secondFactor 31), .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 37), .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.secondFactor 7), .second, .first, (.firstFactor 37), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .second, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 11), (.firstFactor 17), .first, .possible, .first, (.firstFactor 19), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .possible, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .second, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 11), (.secondFactor 29), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .second, .first, .first, (.secondFactor 31), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row10453_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10453 row10453_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10453_block_006
  · decide
  · rfl

def row10453_block_007 : List Nat := [12547, 12553, 12823, 12973, 13411, 13723, 14461]
def plan10453_block_007 : List Budget.Choice := [.first, (.firstFactor 11), .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 7), .possible, .possible, .first, (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 13), (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .possible, (.firstFactor 11), (.secondFactor 17), (.firstFactor 13), .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .second, .possible, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 29), .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .possible, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .second, .first, (.secondFactor 37), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .possible, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, .second, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .possible]
set_option maxRecDepth 2048 in
theorem row10453_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10453 row10453_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10453_block_007
  · decide
  · rfl

def row10453_block_008 : List Nat := [15451]
def plan10453_block_008 : List Budget.Choice := [(.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .second, .second, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 13), .first, (.secondFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 19), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 23), (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .possible, .first, .first, .first, (.secondFactor 17), .first, .second, .first, (.secondFactor 7), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10453_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10453 row10453_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10453_block_008
  · decide
  · rfl

def row10453_block_009 : List Nat := []
theorem row10453_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10453 row10453_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10453_block_010 : List Nat := []
theorem row10453_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10453 row10453_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10453_block_011 : List Nat := []
theorem row10453_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10453 row10453_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10453_block_012 : List Nat := []
theorem row10453_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10453 row10453_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10453_block_013 : List Nat := []
theorem row10453_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10453 row10453_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10453_large_left_left : List Nat := row10453_small ++ row10453_block_005
theorem row10453_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10453 row10453_large_left_left :=
  Budget.RowUpperBound.branch row10453_small_covers row10453_block_005_covers

def row10453_large_left_right_right : List Nat := row10453_block_007 ++ row10453_block_008
theorem row10453_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10453 row10453_large_left_right_right :=
  Budget.RowUpperBound.branch row10453_block_007_covers row10453_block_008_covers

def row10453_large_left_right : List Nat := row10453_block_006 ++ row10453_large_left_right_right
theorem row10453_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10453 row10453_large_left_right :=
  Budget.RowUpperBound.branch row10453_block_006_covers row10453_large_left_right_right_covers

def row10453_large_left : List Nat := row10453_large_left_left ++ row10453_large_left_right
theorem row10453_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10453 row10453_large_left :=
  Budget.RowUpperBound.branch row10453_large_left_left_covers row10453_large_left_right_covers

def row10453_large_right_left : List Nat := row10453_block_009 ++ row10453_block_010
theorem row10453_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10453 row10453_large_right_left :=
  Budget.RowUpperBound.branch row10453_block_009_covers row10453_block_010_covers

def row10453_large_right_right_right : List Nat := row10453_block_012 ++ row10453_block_013
theorem row10453_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10453 row10453_large_right_right_right :=
  Budget.RowUpperBound.branch row10453_block_012_covers row10453_block_013_covers

def row10453_large_right_right : List Nat := row10453_block_011 ++ row10453_large_right_right_right
theorem row10453_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10453 row10453_large_right_right :=
  Budget.RowUpperBound.branch row10453_block_011_covers row10453_large_right_right_right_covers

def row10453_large_right : List Nat := row10453_large_right_left ++ row10453_large_right_right
theorem row10453_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10453 row10453_large_right :=
  Budget.RowUpperBound.branch row10453_large_right_left_covers row10453_large_right_right_covers

def row10453_large : List Nat := row10453_large_left ++ row10453_large_right
theorem row10453_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10453 row10453_large :=
  Budget.RowUpperBound.branch row10453_large_left_covers row10453_large_right_covers

def row10453 : List Nat := row10453_large
theorem row10453_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10453 row10453 := row10453_large_covers

end ProjectEulerSolutions.PrimePairLarge
