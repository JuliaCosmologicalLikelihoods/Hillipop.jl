using ChainRulesCore
using Mooncake

# Rule for xspectra_to_xfreq_unnormed
function ChainRulesCore.rrule(::typeof(Hillipop.xspectra_to_xfreq_unnormed), Rspec::AbstractMatrix, weights::AbstractMatrix, xspec2xfreq::Vector{Int}, nxfreq::Int)
    y = Hillipop.xspectra_to_xfreq_unnormed(Rspec, weights, xspec2xfreq, nxfreq)
    
    function pb_xspectra(dy)
        dxcl = dy[1]
        dRspec = zero(Rspec)
        nxspec, lmax1 = size(Rspec)
        
        @inbounds for l in 1:lmax1
            for xs in 1:nxspec
                xf = xspec2xfreq[xs]
                w = weights[xs, l]
                if isfinite(w) && w > 0.0
                    dRspec[xs, l] += dxcl[xf, l] * w
                end
            end
        end
        
        return ChainRulesCore.NoTangent(), dRspec, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
    end
    
    return y, pb_xspectra
end

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(Hillipop.xspectra_to_xfreq_unnormed), Matrix{Float64}, Matrix{Float64}, Vector{Int}, Int}

# Rule for select_spectra
function ChainRulesCore.rrule(::typeof(Hillipop.select_spectra), xcl::AbstractMatrix, lmins::Vector{Int}, lmaxs::Vector{Int}, nxfreq::Int, xspec2xfreq::Vector{Int})
    y = Hillipop.select_spectra(xcl, lmins, lmaxs, nxfreq, xspec2xfreq)
    
    function pb_select_spectra(dy)
        dxcl = zero(xcl)
        offset = 0
        
        @inbounds for xf in 1:nxfreq
            xs_rep = findfirst(==(xf), xspec2xfreq)
            lmin = lmins[xs_rep]
            lmax = lmaxs[xs_rep]
            len = lmax - lmin + 1
            
            @views dxcl[xf, lmin+1:lmax+1] .+= dy[offset+1 : offset+len]
            offset += len
        end
        
        return ChainRulesCore.NoTangent(), dxcl, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
    end
    
    return y, pb_select_spectra
end

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(Hillipop.select_spectra), Matrix{Float64}, Vector{Int}, Vector{Int}, Int, Vector{Int}}
