import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12157_small : List Nat := []
theorem row12157_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12157 row12157_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12157_block_005 : List Nat := []
theorem row12157_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12157 row12157_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12157_block_006 : List Nat := [12241]
def plan12157_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .second, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row12157_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12157 row12157_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12157_block_006
  · decide
  · rfl

def row12157_block_007 : List Nat := [12601, 13687, 13711]
def plan12157_block_007 : List Budget.Choice := [(.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 13), .second, (.firstFactor 7), .possible, .first, (.secondFactor 7), (.firstFactor 11), .second, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 37), .first, (.firstFactor 17), (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 17), (.firstFactor 7), (.firstFactor 19), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 31), (.firstFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, .second, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 19), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .possible, .first, .second, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12157_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12157 row12157_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12157_block_007
  · decide
  · rfl

def row12157_block_008 : List Nat := []
theorem row12157_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12157 row12157_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12157_block_009 : List Nat := []
theorem row12157_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12157 row12157_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12157_block_010 : List Nat := []
theorem row12157_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12157 row12157_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12157_block_011 : List Nat := []
theorem row12157_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12157 row12157_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12157_block_012 : List Nat := []
theorem row12157_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12157 row12157_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12157_block_013 : List Nat := []
theorem row12157_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12157 row12157_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12157_large_left_left : List Nat := row12157_small ++ row12157_block_005
theorem row12157_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12157 row12157_large_left_left :=
  Budget.RowUpperBound.branch row12157_small_covers row12157_block_005_covers

def row12157_large_left_right_right : List Nat := row12157_block_007 ++ row12157_block_008
theorem row12157_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12157 row12157_large_left_right_right :=
  Budget.RowUpperBound.branch row12157_block_007_covers row12157_block_008_covers

def row12157_large_left_right : List Nat := row12157_block_006 ++ row12157_large_left_right_right
theorem row12157_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12157 row12157_large_left_right :=
  Budget.RowUpperBound.branch row12157_block_006_covers row12157_large_left_right_right_covers

def row12157_large_left : List Nat := row12157_large_left_left ++ row12157_large_left_right
theorem row12157_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12157 row12157_large_left :=
  Budget.RowUpperBound.branch row12157_large_left_left_covers row12157_large_left_right_covers

def row12157_large_right_left : List Nat := row12157_block_009 ++ row12157_block_010
theorem row12157_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12157 row12157_large_right_left :=
  Budget.RowUpperBound.branch row12157_block_009_covers row12157_block_010_covers

def row12157_large_right_right_right : List Nat := row12157_block_012 ++ row12157_block_013
theorem row12157_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12157 row12157_large_right_right_right :=
  Budget.RowUpperBound.branch row12157_block_012_covers row12157_block_013_covers

def row12157_large_right_right : List Nat := row12157_block_011 ++ row12157_large_right_right_right
theorem row12157_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12157 row12157_large_right_right :=
  Budget.RowUpperBound.branch row12157_block_011_covers row12157_large_right_right_right_covers

def row12157_large_right : List Nat := row12157_large_right_left ++ row12157_large_right_right
theorem row12157_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12157 row12157_large_right :=
  Budget.RowUpperBound.branch row12157_large_right_left_covers row12157_large_right_right_covers

def row12157_large : List Nat := row12157_large_left ++ row12157_large_right
theorem row12157_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12157 row12157_large :=
  Budget.RowUpperBound.branch row12157_large_left_covers row12157_large_right_covers

def row12157 : List Nat := row12157_large
theorem row12157_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12157 row12157 := row12157_large_covers

def row12161_small : List Nat := []
theorem row12161_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12161 row12161_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12161_block_005 : List Nat := []
theorem row12161_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12161 row12161_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12161_block_006 : List Nat := []
def plan12161_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 37), (.secondFactor 23), .first, (.firstFactor 23), .first, .first, (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 19)]
set_option maxRecDepth 2048 in
theorem row12161_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12161 row12161_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12161_block_006
  · decide
  · rfl

def row12161_block_007 : List Nat := [13001, 13121, 13163, 13337]
def plan12161_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 13), (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 29), (.firstFactor 11), (.secondFactor 23), .first, .first, .first, (.secondFactor 19), .possible, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 31), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .possible, .first, .first, (.firstFactor 11), .first, .possible, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 29), (.secondFactor 7), .first, (.firstFactor 19), (.secondFactor 13), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .second, .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12161_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12161 row12161_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12161_block_007
  · decide
  · rfl

def row12161_block_008 : List Nat := []
theorem row12161_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12161 row12161_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12161_block_009 : List Nat := []
theorem row12161_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12161 row12161_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12161_block_010 : List Nat := []
theorem row12161_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12161 row12161_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12161_block_011 : List Nat := []
theorem row12161_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12161 row12161_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12161_block_012 : List Nat := []
theorem row12161_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12161 row12161_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12161_block_013 : List Nat := []
theorem row12161_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12161 row12161_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12161_large_left_left : List Nat := row12161_small ++ row12161_block_005
theorem row12161_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12161 row12161_large_left_left :=
  Budget.RowUpperBound.branch row12161_small_covers row12161_block_005_covers

def row12161_large_left_right_right : List Nat := row12161_block_007 ++ row12161_block_008
theorem row12161_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12161 row12161_large_left_right_right :=
  Budget.RowUpperBound.branch row12161_block_007_covers row12161_block_008_covers

def row12161_large_left_right : List Nat := row12161_block_006 ++ row12161_large_left_right_right
theorem row12161_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12161 row12161_large_left_right :=
  Budget.RowUpperBound.branch row12161_block_006_covers row12161_large_left_right_right_covers

def row12161_large_left : List Nat := row12161_large_left_left ++ row12161_large_left_right
theorem row12161_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12161 row12161_large_left :=
  Budget.RowUpperBound.branch row12161_large_left_left_covers row12161_large_left_right_covers

def row12161_large_right_left : List Nat := row12161_block_009 ++ row12161_block_010
theorem row12161_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12161 row12161_large_right_left :=
  Budget.RowUpperBound.branch row12161_block_009_covers row12161_block_010_covers

def row12161_large_right_right_right : List Nat := row12161_block_012 ++ row12161_block_013
theorem row12161_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12161 row12161_large_right_right_right :=
  Budget.RowUpperBound.branch row12161_block_012_covers row12161_block_013_covers

def row12161_large_right_right : List Nat := row12161_block_011 ++ row12161_large_right_right_right
theorem row12161_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12161 row12161_large_right_right :=
  Budget.RowUpperBound.branch row12161_block_011_covers row12161_large_right_right_right_covers

def row12161_large_right : List Nat := row12161_large_right_left ++ row12161_large_right_right
theorem row12161_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12161 row12161_large_right :=
  Budget.RowUpperBound.branch row12161_large_right_left_covers row12161_large_right_right_covers

def row12161_large : List Nat := row12161_large_left ++ row12161_large_right
theorem row12161_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12161 row12161_large :=
  Budget.RowUpperBound.branch row12161_large_left_covers row12161_large_right_covers

def row12161 : List Nat := row12161_large
theorem row12161_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12161 row12161 := row12161_large_covers

def row12163_small : List Nat := []
theorem row12163_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12163 row12163_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12163_block_005 : List Nat := []
theorem row12163_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12163 row12163_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12163_block_006 : List Nat := []
def plan12163_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 23), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row12163_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12163 row12163_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12163_block_006
  · decide
  · rfl

def row12163_block_007 : List Nat := [12457, 12739, 13171, 13591]
def plan12163_block_007 : List Budget.Choice := [(.secondFactor 37), (.firstFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 17), .first, (.firstFactor 7), .first, .second, .second, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .possible, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 23), (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 19), (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), (.secondFactor 29), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, .first, (.secondFactor 17), (.firstFactor 19), .first, (.firstFactor 23), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 17), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12163_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12163 row12163_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12163_block_007
  · decide
  · rfl

def row12163_block_008 : List Nat := []
theorem row12163_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12163 row12163_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12163_block_009 : List Nat := []
theorem row12163_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12163 row12163_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12163_block_010 : List Nat := []
theorem row12163_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12163 row12163_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12163_block_011 : List Nat := []
theorem row12163_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12163 row12163_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12163_block_012 : List Nat := []
theorem row12163_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12163 row12163_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12163_block_013 : List Nat := []
theorem row12163_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12163 row12163_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12163_large_left_left : List Nat := row12163_small ++ row12163_block_005
theorem row12163_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12163 row12163_large_left_left :=
  Budget.RowUpperBound.branch row12163_small_covers row12163_block_005_covers

def row12163_large_left_right_right : List Nat := row12163_block_007 ++ row12163_block_008
theorem row12163_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12163 row12163_large_left_right_right :=
  Budget.RowUpperBound.branch row12163_block_007_covers row12163_block_008_covers

def row12163_large_left_right : List Nat := row12163_block_006 ++ row12163_large_left_right_right
theorem row12163_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12163 row12163_large_left_right :=
  Budget.RowUpperBound.branch row12163_block_006_covers row12163_large_left_right_right_covers

def row12163_large_left : List Nat := row12163_large_left_left ++ row12163_large_left_right
theorem row12163_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12163 row12163_large_left :=
  Budget.RowUpperBound.branch row12163_large_left_left_covers row12163_large_left_right_covers

def row12163_large_right_left : List Nat := row12163_block_009 ++ row12163_block_010
theorem row12163_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12163 row12163_large_right_left :=
  Budget.RowUpperBound.branch row12163_block_009_covers row12163_block_010_covers

def row12163_large_right_right_right : List Nat := row12163_block_012 ++ row12163_block_013
theorem row12163_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12163 row12163_large_right_right_right :=
  Budget.RowUpperBound.branch row12163_block_012_covers row12163_block_013_covers

def row12163_large_right_right : List Nat := row12163_block_011 ++ row12163_large_right_right_right
theorem row12163_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12163 row12163_large_right_right :=
  Budget.RowUpperBound.branch row12163_block_011_covers row12163_large_right_right_right_covers

def row12163_large_right : List Nat := row12163_large_right_left ++ row12163_large_right_right
theorem row12163_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12163 row12163_large_right :=
  Budget.RowUpperBound.branch row12163_large_right_left_covers row12163_large_right_right_covers

def row12163_large : List Nat := row12163_large_left ++ row12163_large_right
theorem row12163_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12163 row12163_large :=
  Budget.RowUpperBound.branch row12163_large_left_covers row12163_large_right_covers

def row12163 : List Nat := row12163_large
theorem row12163_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12163 row12163 := row12163_large_covers

def row12197_small : List Nat := []
theorem row12197_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12197 row12197_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12197_block_005 : List Nat := []
theorem row12197_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12197 row12197_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12197_block_006 : List Nat := []
def plan12197_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 19), .first, (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row12197_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12197 row12197_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12197_block_006
  · decide
  · rfl

def row12197_block_007 : List Nat := [12569, 13367, 13499]
def plan12197_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 29), (.firstFactor 11), (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .second, .first, .first, (.firstFactor 19), (.secondFactor 31), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 11), (.secondFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 7), .possible, .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.secondFactor 37), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12197_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12197 row12197_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12197_block_007
  · decide
  · rfl

def row12197_block_008 : List Nat := []
theorem row12197_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12197 row12197_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12197_block_009 : List Nat := []
theorem row12197_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12197 row12197_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12197_block_010 : List Nat := []
theorem row12197_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12197 row12197_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12197_block_011 : List Nat := []
theorem row12197_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12197 row12197_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12197_block_012 : List Nat := []
theorem row12197_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12197 row12197_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12197_block_013 : List Nat := []
theorem row12197_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12197 row12197_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12197_large_left_left : List Nat := row12197_small ++ row12197_block_005
theorem row12197_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12197 row12197_large_left_left :=
  Budget.RowUpperBound.branch row12197_small_covers row12197_block_005_covers

def row12197_large_left_right_right : List Nat := row12197_block_007 ++ row12197_block_008
theorem row12197_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12197 row12197_large_left_right_right :=
  Budget.RowUpperBound.branch row12197_block_007_covers row12197_block_008_covers

def row12197_large_left_right : List Nat := row12197_block_006 ++ row12197_large_left_right_right
theorem row12197_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12197 row12197_large_left_right :=
  Budget.RowUpperBound.branch row12197_block_006_covers row12197_large_left_right_right_covers

def row12197_large_left : List Nat := row12197_large_left_left ++ row12197_large_left_right
theorem row12197_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12197 row12197_large_left :=
  Budget.RowUpperBound.branch row12197_large_left_left_covers row12197_large_left_right_covers

def row12197_large_right_left : List Nat := row12197_block_009 ++ row12197_block_010
theorem row12197_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12197 row12197_large_right_left :=
  Budget.RowUpperBound.branch row12197_block_009_covers row12197_block_010_covers

def row12197_large_right_right_right : List Nat := row12197_block_012 ++ row12197_block_013
theorem row12197_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12197 row12197_large_right_right_right :=
  Budget.RowUpperBound.branch row12197_block_012_covers row12197_block_013_covers

def row12197_large_right_right : List Nat := row12197_block_011 ++ row12197_large_right_right_right
theorem row12197_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12197 row12197_large_right_right :=
  Budget.RowUpperBound.branch row12197_block_011_covers row12197_large_right_right_right_covers

def row12197_large_right : List Nat := row12197_large_right_left ++ row12197_large_right_right
theorem row12197_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12197 row12197_large_right :=
  Budget.RowUpperBound.branch row12197_large_right_left_covers row12197_large_right_right_covers

def row12197_large : List Nat := row12197_large_left ++ row12197_large_right
theorem row12197_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12197 row12197_large :=
  Budget.RowUpperBound.branch row12197_large_left_covers row12197_large_right_covers

def row12197 : List Nat := row12197_large
theorem row12197_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12197 row12197 := row12197_large_covers

end ProjectEulerSolutions.PrimePairLarge
